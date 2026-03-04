from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


BAD_PATTERNS = [
    r"\bnot assessed\b",
    r"\bassessed\b",
    r"\bscore\b",
    r"\bresult\b",
    r"\bdetected\b",
    r"\bnot detected\b",
    r"\bbiopsy\b",
    r"\bcharacteristic\b",
    r"\bcenter of town\b",
    r"\btherapy\b",
    r"\bprocedure\b",
    r"\bmeasurement\b",
    r"\bpercent\b",
    r"\bmg\b",
    r"\bmmhg\b",
    r"\bcm\b",
]
BAD_RE = re.compile("|".join(BAD_PATTERNS), flags=re.IGNORECASE)

def is_plausible_symptom(name: str) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip()
    if len(s) < 3 or len(s) > 40:
        return False
    if BAD_RE.search(s):
        return False
    # exclude too many non-letters
    letters = sum(ch.isalpha() for ch in s)
    if letters / max(1, len(s)) < 0.65:
        return False
    # exclude parenthetical lab-like concepts
    if "(" in s or ")" in s:
        return False
    return True


def main(
    project_root: str = ".",
    n_queries: int = 120,
    seed: int = 7,
    min_symptoms: int = 2,
    max_symptoms: int = 4,
    out_path: str = "eval/queries_silver.jsonl",
):
    random.seed(seed)
    root = Path(project_root).resolve()
    parsed = root / "data" / "parsed"

    edge_evid_path = parsed / "graph_edge_evidence.parquet"
    concepts_path = parsed / "graph_concepts.parquet"

    edge = pd.read_parquet(edge_evid_path)[
        ["pmcid", "symptom_concept_id", "evidence_score"]
    ].copy()

    concepts = pd.read_parquet(concepts_path)[["concept_id", "canonical_name", "concept_types"]].copy()
    concepts["concept_types"] = concepts["concept_types"].astype(str)

    symptom_map = (
        concepts[concepts["concept_types"].str.contains("SYMPTOM", na=False)]
        .set_index("concept_id")["canonical_name"]
        .to_dict()
    )

    # Filter symptom_map by plausibility
    symptom_map = {k: v for k, v in symptom_map.items() if is_plausible_symptom(v)}

    edge = edge[edge["symptom_concept_id"].isin(symptom_map.keys())].copy()
    edge = edge.dropna(subset=["pmcid", "symptom_concept_id"])

    per = (
        edge.groupby(["pmcid", "symptom_concept_id"], as_index=False)
        .agg(max_score=("evidence_score", "max"))
        .sort_values(["pmcid", "max_score"], ascending=[True, False])
    )

    counts = per.groupby("pmcid")["symptom_concept_id"].nunique()
    eligible_pmcids = counts[counts >= min_symptoms].index.tolist()
    if not eligible_pmcids:
        raise RuntimeError(f"No eligible PMCIDs with >= {min_symptoms} plausible symptoms found.")

    random.shuffle(eligible_pmcids)
    chosen = eligible_pmcids[: n_queries * 3]  # over-sample because we may skip

    out_rows: List[Dict] = []
    used_pmcids = set()

    for pmcid in chosen:
        if pmcid in used_pmcids:
            continue

        sub = per[per["pmcid"] == pmcid].head(40)
        symptom_ids = sub["symptom_concept_id"].tolist()

        k = random.randint(min_symptoms, max_symptoms)
        picked_ids = symptom_ids[: max(k, min_symptoms)]
        if len(picked_ids) > k:
            picked_ids = random.sample(picked_ids, k)

        names = [symptom_map.get(sid, "") for sid in picked_ids]
        names = [n for n in names if is_plausible_symptom(n)]

        if len(names) < min_symptoms:
            continue

        used_pmcids.add(pmcid)
        out_rows.append({
            "qid": f"silver_{len(out_rows)+1:04d}",
            "query": ", ".join(names),
            "relevant_pmcids": [str(pmcid)],
        })

        if len(out_rows) >= n_queries:
            break

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} queries to {out_file}")
    print("First 5 examples:")
    for r in out_rows[:5]:
        print(r)


if __name__ == "__main__":
    main()
