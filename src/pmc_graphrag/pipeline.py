from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

import spacy
import scispacy  # noqa: F401


Mode = Literal["graph", "vector", "hybrid"]


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


@dataclass
class RetrievalResult:
    query: str
    mode: Mode
    symptoms: List[Tuple[str, float, str]]  # (concept_id, score, surface_text)
    ranked_conditions_df: pd.DataFrame
    context: Dict
    latency_sec: float


class GraphRAGPipeline:
    """
    Notebook-05 compatible pipeline wrapper.

    Exposes:
      - retrieve(query, mode="graph"|"vector"|"hybrid")
      - retrieved_pmcids_from_context(context)

    Uses your existing artifacts under:
      <project_root>/data/parsed/
      <project_root>/data/index/
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        spacy_model: str = "en_core_sci_sm",
    ):
        self.project_root = (project_root or Path(".")).resolve()
        self.parsed_dir = self.project_root / "data" / "parsed"
        self.index_dir = self.project_root / "data" / "index"

        # Notebook 05 inputs
        self.in_chunks = self.parsed_dir / "pmc_retrieval_candidates.parquet"
        self.in_concepts = self.parsed_dir / "graph_concepts.parquet"
        self.in_edge_evid = self.parsed_dir / "graph_edge_evidence.parquet"
        self.in_edges_sc = self.parsed_dir / "graph_edges_symptom_condition.parquet"

        self.faiss_index_path = self.index_dir / "chunks.faiss"
        self.lookup_parquet = self.index_dir / "chunk_lookup.parquet"
        self.meta_json = self.index_dir / "chunk_meta.json"

        self._load_artifacts()
        self._load_spacy(spacy_model)
        self._build_known_symptoms()

    # -------------------------
    # Loading
    # -------------------------
    def _load_artifacts(self) -> None:
        for p in [
            self.in_chunks, self.in_concepts, self.in_edge_evid, self.in_edges_sc,
            self.faiss_index_path, self.lookup_parquet, self.meta_json
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required artifact: {p}")

        self.chunks_df = pd.read_parquet(self.in_chunks)[
            ["chunk_id", "pmcid", "article_title", "section", "chunk_text", "token_count", "license"]
        ]
        self.concepts_df = pd.read_parquet(self.in_concepts)
        self.edge_evid_df = pd.read_parquet(self.in_edge_evid)
        self.edges_sc_df = pd.read_parquet(self.in_edges_sc)

        self.lookup_df = pd.read_parquet(self.lookup_parquet)
        self.meta = json.loads(self.meta_json.read_text(encoding="utf-8"))

        self.name_map = self.concepts_df.set_index("concept_id")["canonical_name"].to_dict()

        # For vector->condition aggregation (Notebook 05)
        self.edge_evid_by_chunk = self.edge_evid_df[[
            "chunk_id", "condition_concept_id", "evidence_score", "pmcid"
        ]]

        # Embedder + FAISS
        self.embedder = SentenceTransformer(self.meta["embedding_model"])
        self.index = faiss.read_index(str(self.faiss_index_path))
        if self.index.d != self.meta["embedding_dim"]:
            raise ValueError(
                f"FAISS dim mismatch: index.d={self.index.d} vs meta={self.meta['embedding_dim']}"
            )

    def _load_spacy(self, spacy_model: str) -> None:
        # Notebook 05 uses lightweight NER only, no UMLS linker at inference.
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load spaCy model '{spacy_model}'. "
                f"Install it in your conda env (example): "
                f"python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            ) from e

    def _build_known_symptoms(self) -> None:
        # Notebook 05 alias strategy
        ALIASES = {
            "low blood pressure": "hypotension",
            "confusion": "altered mental status",
            "mental confusion": "altered mental status",
            "disorientation": "altered mental status",
            "infection": "infection",
        }

        symptom_concepts = self.concepts_df[
            self.concepts_df["concept_types"].astype(str).str.contains("SYMPTOM")
        ].copy()

        known_symptoms: Dict[str, str] = {}
        for _, r in symptom_concepts.iterrows():
            name = r.get("canonical_name")
            if not isinstance(name, str) or not name.strip():
                continue

            canon = normalize_text(name)
            known_symptoms[canon] = r["concept_id"]

            # inject aliases that map to this canonical symptom
            for alias, target in ALIASES.items():
                if target in canon:
                    known_symptoms[normalize_text(alias)] = r["concept_id"]

        self.known_symptoms = known_symptoms

    # -------------------------
    # Notebook 05 logic
    # -------------------------
    def extract_symptom_concepts(
        self,
        query: str,
    ) -> List[Tuple[str, float, str]]:
        """
        Returns (concept_id, score, surface_text)
        Ensures surface_text is never empty.
        """
        doc = self.nlp(query)
        hits = {}

        q_norm = normalize_text(query)

        # 1) substring match over full query
        for k, cid in self.known_symptoms.items():
            if k and k in q_norm:
                hits[cid] = (1.0, k)

        # 2) NER span match (preferred surface form)
        for ent in doc.ents:
            ent_norm = normalize_text(ent.text)
            if ent_norm in self.known_symptoms:
                cid = self.known_symptoms[ent_norm]
                hits[cid] = (1.0, ent.text)

        # drop any empty surface texts
        return [
            (cid, sc, txt)
            for cid, (sc, txt) in hits.items()
            if isinstance(txt, str) and txt.strip()
        ]

    def graph_first_candidates(
        self,
        symptom_ids: List[str],
        top_k: int = 40,
    ) -> pd.DataFrame:
        """
        Graph-first retrieval with clinical symptom weighting.
        Fever is downweighted because it is non-specific.
        """
        if not symptom_ids:
            return pd.DataFrame(columns=["condition_concept_id", "graph_score", "graph_support"])

        # Clinical symptom weights (Notebook 05)
        SYMPTOM_WEIGHTS = {
            "UMLS:C0015967": 0.3,  # fever (non-specific)
            "UMLS:C4304283": 1.0,  # hypotension
            "UMLS:C5187432": 1.0,  # altered mental status
        }

        sub = self.edges_sc_df[self.edges_sc_df["symptom_concept_id"].isin(symptom_ids)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["condition_concept_id", "graph_score", "graph_support"])

        sub["symptom_weight"] = sub["symptom_concept_id"].map(
            lambda cid: SYMPTOM_WEIGHTS.get(cid, 1.0)
        )
        sub["weighted_graph_score"] = sub["weighted_score"] * sub["symptom_weight"]

        return (
            sub.groupby("condition_concept_id", as_index=False)
            .agg(
                graph_score=("weighted_graph_score", "sum"),
                graph_support=("support_count", "sum"),
            )
            .sort_values(["graph_score", "graph_support"], ascending=False)
            .head(top_k)
        )

    def faiss_chunks(self, query: str, k: int = 30) -> pd.DataFrame:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q, k)
        out = self.lookup_df.iloc[idxs[0]].copy()
        out["similarity"] = scores[0]
        return out

    def embedding_candidates(self, query: str, k_chunks: int = 30, k_cond: int = 40) -> pd.DataFrame:
        chunks = self.faiss_chunks(query, k_chunks)
        sub = self.edge_evid_by_chunk[self.edge_evid_by_chunk["chunk_id"].isin(chunks["chunk_id"])]
        if sub.empty:
            return pd.DataFrame(columns=["condition_concept_id", "emb_score", "emb_support"])

        return (
            sub.groupby("condition_concept_id", as_index=False)
            .agg(
                emb_score=("evidence_score", "sum"),
                emb_support=("pmcid", "nunique"),
            )
            .sort_values(["emb_score", "emb_support"], ascending=False)
            .head(k_cond)
        )

    @staticmethod
    def log1p(x: float) -> float:
        return math.log1p(max(0.0, float(x)))

    def merge_rerank(self, graph_df: pd.DataFrame, emb_df: pd.DataFrame, use_graph: bool, top_n: int = 5) -> pd.DataFrame:
        merged = pd.merge(
            graph_df, emb_df,
            on="condition_concept_id",
            how="outer"
        )
        merged = merged.infer_objects(copy=False).fillna(0.0)

        # Dynamic weighting based on symptom specificity
        if use_graph:
            W_GRAPH, W_EMB = 0.8, 0.2
        else:
            W_GRAPH, W_EMB = 0.4, 0.6

        merged["final_score"] = (
            W_GRAPH * merged["graph_score"].apply(self.log1p)
            + W_EMB * merged["emb_score"].apply(self.log1p)
        )

        return merged.sort_values(
            ["final_score", "graph_support", "emb_support"],
            ascending=False
        ).head(top_n)

    def select_evidence(
        self,
        condition_id: str,
        symptom_ids: List[str],
        max_snips: int = 6,
    ) -> pd.DataFrame:
        sub = self.edge_evid_df[self.edge_evid_df["condition_concept_id"] == condition_id]

        if symptom_ids:
            sub2 = sub[sub["symptom_concept_id"].isin(symptom_ids)]
            if not sub2.empty:
                sub = sub2

        picked = []
        seen = defaultdict(int)

        for _, r in sub.sort_values("evidence_score", ascending=False).iterrows():
            if seen[r["pmcid"]] >= 2:
                continue
            picked.append(r)
            seen[r["pmcid"]] += 1
            if len(picked) >= max_snips:
                break

        return pd.DataFrame(picked)

    def build_context(
        self,
        query: str,
        symptoms: List[Tuple[str, float, str]],
        ranked: pd.DataFrame,
    ) -> Dict:
        conditions = []
        for _, r in ranked.iterrows():
            ev = self.select_evidence(r["condition_concept_id"], [s[0] for s in symptoms])
            conditions.append({
                "concept_id": r["condition_concept_id"],
                "label": self.name_map.get(r["condition_concept_id"]),
                "final_score": float(r.get("final_score", 0.0)),
                "evidence": ev[[
                    "pmcid", "article_title", "section", "chunk_id", "snippet", "evidence_score"
                ]].to_dict(orient="records"),
            })

        return {
            "query": query,
            "symptoms": [
                {"concept_id": cid, "label": self.name_map.get(cid), "mention": txt}
                for cid, _, txt in symptoms
            ],
            "conditions": conditions,
        }

    def retrieve(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_conditions: int = 5,
        top_graph: int = 40,
        top_emb_chunks: int = 30,
        top_emb_cond: int = 40,
    ) -> RetrievalResult:
        import time

        t0 = time.perf_counter()

        symptoms = self.extract_symptom_concepts(query)

        # Notebook 05: drop generic infection
        GENERIC_SYMPTOMS = {"UMLS:C5139432"}
        symptoms = [s for s in symptoms if s[0] not in GENERIC_SYMPTOMS]
        symptom_ids = [cid for cid, _, _ in symptoms]

        SPECIFIC_SYMPTOMS = {
            "UMLS:C4304283",  # hypotension
            "UMLS:C5187432",  # altered mental status
        }
        specific_count = sum(1 for cid, _, _ in symptoms if cid in SPECIFIC_SYMPTOMS)
        use_graph = specific_count >= 1

        graph_df = pd.DataFrame(columns=["condition_concept_id", "graph_score", "graph_support"])
        emb_df = pd.DataFrame(columns=["condition_concept_id", "emb_score", "emb_support"])

        if mode in ("graph", "hybrid"):
            graph_df = self.graph_first_candidates(symptom_ids, top_k=top_graph)

        if mode in ("vector", "hybrid"):
            emb_df = self.embedding_candidates(query, k_chunks=top_emb_chunks, k_cond=top_emb_cond)

        if mode == "graph":
            ranked = graph_df.copy()
            ranked["final_score"] = ranked["graph_score"].apply(self.log1p)
            ranked = ranked.sort_values(["final_score", "graph_support"], ascending=False).head(top_conditions)

        elif mode == "vector":
            ranked = emb_df.copy()
            ranked["final_score"] = ranked["emb_score"].apply(self.log1p)
            ranked = ranked.sort_values(["final_score", "emb_support"], ascending=False).head(top_conditions)

        else:
            ranked = self.merge_rerank(graph_df, emb_df, use_graph=use_graph, top_n=top_conditions)

        context = self.build_context(query, symptoms, ranked)
        t1 = time.perf_counter()

        return RetrievalResult(
            query=query,
            mode=mode,
            symptoms=symptoms,
            ranked_conditions_df=ranked.reset_index(drop=True),
            context=context,
            latency_sec=(t1 - t0),
        )

    @staticmethod
    def retrieved_pmcids_from_context(context: Dict, max_docs: int = 50) -> List[str]:
        out: List[str] = []
        seen = set()
        for cond in context.get("conditions", []):
            for ev in cond.get("evidence", []):
                pmcid = ev.get("pmcid")
                if pmcid and pmcid not in seen:
                    out.append(str(pmcid))
                    seen.add(pmcid)
                if len(out) >= max_docs:
                    return out
        return out
