# PMC_GraphRAG — Graph-First Biomedical Retrieval with Citation-Grounded Evidence

This repository contains a biomedical literature retrieval system built over the PMC Open Access Subset. Given a free-text symptom-style query, it retrieves citation-backed evidence from peer-reviewed articles using:

* **Graph traversal** over extracted relationships (Symptom → Condition → Evidence → Article)
* **Semantic similarity search** over chunk embeddings (FAISS) as a fallback recall path

*A quick note upfront: this is an applied NLP / information retrieval project. It is not a diagnostic tool and it does not use any patient records, EHRs, or protected health information.*

## Why build this?
Naive RAG on biomedical text often returns vaguely relevant chunks without preserving *why* they are relevant, or how they connect to a symptom query in a traceable way. I wanted a retrieval layer that is:

* **inspectable** (you can see the path from symptoms → conditions → evidence)
* **citation-safe** (every snippet is linked back to a PMCID + section + chunk)
* **explainable** (the graph provides explicit relational structure)

The high-level idea is: push heavy concept linking offline, then do lightweight symptom grounding at query time, and always keep evidence provenance intact.

## What the system does

### Included
* Ingests PMC OA articles with explicit open licenses (CC0/CC-BY/CC-BY-SA/CC-BY-ND).
* Parses messy JATS XML into structured text.
* Restricts retrieval to Results and Discussion sections.
* Extracts and links mentions to UMLS concepts offline (SciSpaCy).
* Builds a provenance-preserving evidence graph: Symptom ↔ Condition ↔ Evidence Chunk ↔ Article.
* Builds a local FAISS index over chunk embeddings for semantic retrieval.
* Retrieves evidence at query time via:
    * a graph-first path (concept traversal)
    * a vector fallback path (semantic recall)
* Optional: uses an LLM only as a constrained writer over retrieved evidence (not as a primary reasoning engine).

### Explicitly excluded
* No patient data / PHI.
* No EHR or MIMIC data.
* No diagnosis or treatment planning.
* No model training on clinical outcomes.

## Quantitative snapshot (local run)
From a representative local build:

* **1,000** PMC OA articles requested
* **994** unique PMCIDs parsed (99.4% success)
* **88,995** paragraph-level rows extracted
* **20,838** Results/Discussion rows retained
* **12,935** retrieval-ready evidence chunks
* **5,779** UMLS-grounded concepts
* **609,882** provenance-linked evidence rows
* **290,910** aggregated symptom-condition edges
* **~0.23–0.25 sec** typical retrieval latency (local sanity query suite; LLM step excluded)

*These numbers reflect a local-first, notebook-driven build for rapid iteration, not a hosted production cluster.*

## Core idea: graph-first retrieval with semantic fallback
At inference time, the system runs two retrieval paths:

1.  **Graph path (primary signal):** Map query symptoms to known concepts, traverse symptom→condition edges, and gather supporting evidence chunks and source PMCIDs.
2.  **Embedding path (fallback recall):** Encode the raw query, retrieve semantically similar chunks via FAISS, and map those chunks back to candidate conditions through the evidence tables.
3.  **Merge & rerank (hybrid mode):** Combine candidates from both paths to produce a final condition ranking and evidence pack.

In practice (see evaluation results below), my current implementation behaves more like graph-first retrieval with vector backoff, which is a reasonable outcome for this setting.

## Engineering design notes (the “why” behind a few choices)
* **Offline heavy linking, lightweight online grounding:** Loading the full UmlsEntityLinker at inference caused large memory spikes on my local machine. I moved entity linking offline, and kept the online step lightweight (NER + lexical grounding into already-known concept space). This keeps query-time memory stable.
* **Dense offline graph, prune at query time:** I built a high-recall graph offline and push pruning to retrieval time, while keeping provenance on every edge so citations remain traceable.
* **Token-based chunking:** Biomedical paragraphs vary wildly in length. Token windows (~350 tokens, 50 overlap) produce more consistent embedding behavior than paragraph boundaries.
* **Generic symptom handling:** Some symptoms are poor discriminators (e.g., fever). I added downweighting logic so generic symptoms don’t dominate the condition ranking.

## Pipeline overview
```text
User Query
    ↓
Symptom Extraction (NER + Lexical)
    ↓
┌─────────────────────┬──────────────────┐
│ Graph Traversal     │ FAISS Retrieval  │
│ (UMLS-grounded)     │ (Embeddings)     │
└─────────────────────┴──────────────────┘
            ↓
      Candidate Fusion
            ↓
       Evidence Pack
            ↓
     Optional LLM Output
```

## Notebooks (build pipeline)
* `notebooks/01_download_parse.ipynb` — ingestion + JATS parsing → `pmc_raw_articles.parquet`
* `notebooks/02_sql_filtering.ipynb` — Results/Discussion filtering + chunking → `pmc_retrieval_candidates.parquet`
* `notebooks/03_scispacy_uml.ipynb` — offline concept linking + graph tables (`graph_concepts.parquet`, `graph_mentions.parquet`, `graph_edge_evidence.parquet`, `graph_edges_symptom_condition.parquet`, …)
* `notebooks/04_hybrid_retreival.ipynb` — embeddings + FAISS index (`chunks.faiss`, `chunk_lookup.parquet`, `chunk_meta.json`)
* `notebooks/05_query_compose.ipynb` — end-to-end retrieval + composition

## Lightweight Python wrapper (for evaluation + testing)
`src/pmc_graphrag/pipeline.py`

A notebook-compatible wrapper exposing:
* `GraphRAGPipeline.retrieve(query, mode={graph|vector|hybrid})`
* `retrieved_pmcids_from_context(context)`

## Evaluation (baseline benchmark)
Since I don’t have clinical gold labels, I built a silver retrieval benchmark derived from the corpus itself.

### Silver benchmark construction (weak supervision)
1.  Sample PMCIDs that have multiple symptom-linked evidence rows.
2.  Select 2–4 plausible “symptom-like” concepts linked to that PMCID.
3.  Use those symptom strings as the query.
4.  Treat the original PMCID as the relevant document.

This is not meant to measure “diagnostic correctness.” It measures: *can the system retrieve the source document that generated the symptom evidence?*

The generator lives in: `eval/generate_silver_set.py` → produces `eval/queries_silver.jsonl`

### Metrics and runner
* `eval/metrics.py` (Recall@k, MRR, nDCG@k)
* `eval/run_eval.py` runs graph, vector, hybrid and writes `eval/results.csv`

### Current benchmark results (n=120 queries)
| mode   | p50 latency (s) | p95 latency (s) | Recall@10 | Recall@20 | MRR@50 | nDCG@10 |
|--------|-----------------|-----------------|-----------|-----------|--------|---------|
| graph  | 0.146           | 0.154           | 0.433     | 0.500     | 0.208  | 0.256   |
| hybrid | 0.217           | 0.241           | 0.408     | 0.483     | 0.198  | 0.244   |
| vector | 0.191           | 0.215           | 0.392     | 0.425     | 0.207  | 0.249   |

### Interpretation (honest)
* On this benchmark, graph retrieval is the strongest signal.
* The current hybrid fusion does not consistently improve over graph-only; it is best read as a fallback mechanism rather than a guaranteed booster.
* This is still a meaningful outcome: it suggests the UMLS-grounded structure is providing real retrieval value beyond embeddings alone.

## How to run

### Environment
```bash
conda create -n pmc_graphrag python=3.10
conda activate pmc_graphrag
pip install -r requirements.txt
```
If you plan to use the optional LLM synthesis step:
```bash
pip install python-dotenv
```

### Build the corpus (notebooks)
Run 01 → 05 in order under `notebooks/`.

### Run evaluation (Windows / PowerShell)
This repo includes a small helper to ensure imports resolve cleanly. 

Generate the silver set:
```powershell
python .\eval\generate_silver_set.py
```

Run evaluation (ensure repo root + src are on PYTHONPATH):
```powershell
$env:PYTHONPATH="D:\Pictures\pmc_graphrag;D:\Pictures\pmc_graphrag\src"
python -m eval.run_eval --queries eval\queries_silver.jsonl --mode all
```

The runner prints a summary table and writes: `eval/results.csv`

## Example output (illustrative)
* **Input symptoms:** fever, hypotension, confusion
* **Top literature-associated condition:** Sepsis
* **Evidence snippets:** returned with PMCID + section provenance.

*Note: associations reflect literature co-occurrence, not a diagnosis.*

## Project highlights / impact summary
* Built a provenance-preserving biomedical retrieval pipeline over PMC Open Access articles, transforming raw JATS XML into an evidence layer usable for citation-grounded retrieval.
* Constructed a UMLS-grounded evidence graph linking symptoms → conditions → evidence chunks → source articles, enabling inspectable multi-hop retrieval.
* Implemented a lightweight inference wrapper and evaluation harness to benchmark graph vs embedding vs hybrid retrieval using standard IR metrics (Recall@k, MRR, nDCG).

## Tech stack
* **Data & processing:** Python, Pandas, DuckDB, Parquet
* **NLP & graph:** SciSpaCy, UMLS, NetworkX
* **Search & embeddings:** SentenceTransformers, FAISS
* **Infrastructure:** AWS Open Data access (boto3)
* **Optional LLM:** DeepSeek API (constrained writing only)

## Future work (high-value next steps)
* Tighten the “symptom-like concept” filter to better match real user symptom phrasing (reducing administrative / demographic concepts).
* Add a small manually curated query set (20–50 queries) for a higher-precision benchmark.
* Experiment with alternative fusion strategies (e.g., max fusion / reciprocal rank fusion) and evaluate against the benchmark.
* Package inference as a lightweight API service (FastAPI) with a minimal regression test suite.

## Disclaimer
*This repository is a portfolio project focused on data engineering, NLP, and information retrieval. It is not a medical device, diagnostic system, or clinical decision support tool.*