# PMC_GraphRAG — Hybrid Graph-RAG for Citation-Grounded Biomedical Literature Retrieval

This is a biomedical literature retrieval system I built over the **PMC Open Access Subset**. If you give it a free-text symptom query, it retrieves citation-backed evidence from peer-reviewed articles. It does this by combining:

- **Graph traversal** over extracted relationships (Symptom → Condition → Evidence → Article)
- **Semantic similarity search** over chunk embeddings using FAISS

A quick note upfront: This is an applied NLP and retrieval systems project. It is **not** a diagnostic tool, and it completely avoids using patient records, EHRs, or any protected health information.

---

## Why build this?

Throwing naive RAG at medical text usually results in vaguely relevant chunks being passed to an LLM without preserving *why* they matter or how they actually connect to the user's symptoms. I wanted to build a structured evidence layer instead.

In this pipeline:
- Symptoms and conditions are explicitly extracted from the literature.
- Mentions are linked to UMLS concepts offline.
- Every evidence chunk retains its provenance back to the source article.
- Query-time retrieval merges structured graph signals with semantic recall.

The end result is a retrieval pipeline that is inspectable, citation-safe, and explainable—a step up from a standard vector-search chatbot.

---

## What the system does

### What's included
- Ingests PMC Open Access articles, filtering explicitly for open licenses (CC0/CC-BY/CC-BY-SA/CC-BY-ND).
- Parses messy JATS XML into clean, structured paragraph-level text.
- Restricts retrieval strictly to **Results** and **Discussion** sections.
- Extracts UMLS-grounded symptom and condition mentions offline via SciSpaCy.
- Builds a provenance-preserving graph (Symptom ↔ Condition ↔ Evidence Chunk ↔ Article).
- Builds a local FAISS index for fallback embedding retrieval.
- Combines graph traversal and semantic retrieval at query time.
- (Optional) Passes retrieved evidence to the DeepSeek API to compose a final, constrained response.

### What's explicitly excluded
- No patient data or protected health information (PHI).
- No EHR or MIMIC data.
- No medical diagnosis or treatment planning.
- No model training on clinical outcome labels.

---

## Quantitative snapshot

Metrics from my current local run:
- **1,000** PMC OA articles requested
- **994** unique PMCIDs parsed (**99.4%** success rate)
- **88,995** paragraph-level rows extracted
- **20,838** Results/Discussion paragraph rows retained
- **12,935** retrieval-ready evidence chunks
- **5,779** UMLS-grounded concepts
- **609,882** provenance-linked evidence rows
- **290,910** aggregated symptom-condition edges
- **~0.23–0.25 sec** hybrid retrieval latency (on my local machine, running the sanity-query suite)

*Note: These numbers reflect a local-first, notebook-driven build for rapid iteration, not a hosted production cluster.*

---

## The core idea: Hybrid Retrieval

Here is how the hybrid retrieval actually works under the hood:

1. **The Graph Path:** Map the query symptoms to known concepts, traverse the symptom→condition edges, and pull the supporting evidence chunks and source articles.
2. **The Embedding Path:** Encode the raw query, retrieve semantically similar chunks via FAISS, and map those chunks back to candidate conditions via the evidence tables.
3. **Merge & Rerank:** Combine candidates from both paths, preserving evidence diversity and provenance.

If you use the optional LLM step, the LLM acts purely as a constrained writer over this retrieved evidence, rather than a primary reasoning engine.

---

## Engineering Design & Trade-offs

Building this involved a few key architectural choices to prioritize stability and memory management over generic conversational fluff:

- **Offline High-Recall Graph:** I built the graph to be dense offline to capture broad co-occurrences. This moves the heavy lifting of pruning to query-time rather than making ingestion a bottleneck. Every edge keeps explicit provenance so citations are always traceable.
- **Solving Query-Time Memory Spikes:** Initially, I tried loading the full `UmlsEntityLinker` at inference, but it caused a ~400MB memory spike on my local hardware. To fix this, the query path now just relies on lightweight NER + lexical grounding, which maps cheaply to the concepts I already grounded in the offline graph.
- **Why Hybrid is Necessary:** Strict ontology mapping misses the weird ways users actually phrase symptoms. The FAISS index acts as a high-recall fallback to catch evidence when the semantic mapping fails.
- **Chunking Strategy:** I went with token-based chunking (~350 tokens, 50 overlap) instead of paragraph boundaries. Biomedical articles vary wildly in paragraph length, so token windows keep the embedding models much happier and more consistent.
- **Handling Non-Specific Symptoms:** Symptoms like *fever* show up in practically every disease, making them terrible discriminators. I added logic to downweight generic symptoms so they don't hijack the condition rankings.

---

## Pipeline Overview

```text
## System Architecture (Simplified)

User Query
    ↓
Symptom Extraction (NER + Lexical)
    ↓
┌───────────────┬─────────────────┐
│ Graph Traversal│ FAISS Retrieval │
│ (UMLS-grounded)│ (Embeddings)    │
└───────────────┴─────────────────┘
            ↓
      Merge & Rerank
            ↓
       Evidence Pack
            ↓
     Optional LLM Output
```

### 1) `01_download_parse.ipynb` — Ingestion + XML Parsing
Builds the raw corpus. This is strictly a plumbing notebook—no NLP or retrieval logic here.
- Queries PMC (initially using `sepsis` + license filters).
- Downloads XML/JSON via boto3 and parses the JATS XML (namespace-agnostic).
- Drops retracted articles.
- **Output:** `pmc_raw_articles.parquet`

### 2) `02_sql_filtering.ipynb` — Evidence Layer Construction
Converts raw text into clean chunks using DuckDB, prioritizing evidence quality over volume.
- Keeps latest article versions, filtering down to Results/Discussion.
- Applies the 350-token sliding window and drops tiny tail chunks.
- **Output:** `pmc_retrieval_candidates.parquet`

### 3) `03_scispacy_umls.ipynb` — Offline Graph Prep
Does the heavy NLP lifting to build the offline graph.
- Runs SciSpaCy to extract and link entities to UMLS.
- Filters down to Symptoms/Findings and Conditions/Diseases.
- **Outputs:** `graph_concepts.parquet`, `graph_mentions.parquet`, `graph_edge_evidence.parquet`, `graph_edges_symptom_condition.parquet`, `graph_nx.pkl`, etc.

### 4) `04_hybrid_retrieval.ipynb` — Embeddings & FAISS Index
Sets up the semantic fallback layer.
- Embeds chunks using `all-MiniLM-L6-v2`.
- Builds a FAISS `IndexFlatIP` index for cosine similarity.
- **Outputs:** `chunks.faiss`, `chunk_lookup.parquet`, `chunk_meta.json`

### 5) `05_query_compose.ipynb` — Inference & Composition
Runs end-to-end hybrid retrieval.
- Merges graph and FAISS candidates.
- Applies symptom specificity weighting.
- Optionally calls DeepSeek to format the final citation-ready response.

---

## Why this is different from a standard RAG project

Most RAG tutorials stop at: load text → chunk → embed → retrieve top-k → prompt LLM.

This project takes it further by adding ontology-grounded entity extraction, explicit graph construction, memory-safe online/offline separation, and strict evidence provenance attached to every single retrieval hop. It’s built more like an information retrieval system than a wrapper around an API.

---

## Evaluation Notes (Retrieval Behavior)

Since I don't have clinical gold labels for this subset, I evaluated the system based on retrieval behavior diagnostics:

- **Graph vs Embedding Overlap:** Top-K condition overlap between the graph and FAISS paths varies by query (Jaccard@10 ≈ 0.0–0.6). This is a good thing—it shows the two paths provide complementary signals rather than redundant hits.
- **Hybrid Recall Benefit:** The embedding fallback consistently recovers relevant conditions when the query is underspecified or the symptom grounding is weak.
- **Evidence Diversity:** Final evidence packs usually pull from 8–16 distinct PMCIDs per query, which reduces the chance of a single paper dominating the results.
- **Fallback Behavior:** If a query is incredibly generic and no specific symptoms are extracted, the system degrades gracefully to embedding-only retrieval rather than crashing or returning blank.
- **Latency:** End-to-end retrieval takes about ~0.23–0.25 seconds per query locally (excluding the optional LLM generation).

---

## How to run it

### 1. Set up the environment
```bash
conda create -n pmc-graphrag python=3.10
conda activate pmc-graphrag
pip install -r requirements.txt
pip install python-dotenv # if you plan to use the DeepSeek generation
```

### 2. Add API keys (Optional)
Create a `.env` file in the root if you want the final LLM synthesis:
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 3. Run the pipeline
Execute notebooks `01` through `05` in order. 

If you want to scale up the corpus, just change `MAX_ARTICLES = 1000` in notebook 01 and rerun everything downstream.

---

## Example Output (Illustrative)

**Input symptoms:** `fever, hypotension, confusion`

**Top literature-associated condition:** **Sepsis**

**Supporting evidence (excerpted):**
- **PMC1234567 — Results section** *“…patients presenting with hypotension and altered mental status were significantly more likely to develop septic shock…”*
- **PMC2345678 — Discussion section** *“…fever combined with circulatory instability remains a key clinical indicator associated with sepsis-related outcomes…”*

*Notes: Associations reflect literature co-occurrence, not a diagnosis. Every snippet maps directly to its source PMC Open Access article.*

---

## Project Highlights / Impact Summary

- Built a hybrid Graph-RAG biomedical retrieval pipeline over 1,000 PMC Open Access articles, processing 88K+ paragraphs into 12.9K evidence chunks with a 99.4% ingestion success rate.
- Constructed an explainable knowledge graph containing 5.8K UMLS-grounded concepts and 609K provenance-linked evidence rows, allowing multi-hop traversal between symptoms, conditions, and citations.
- Engineered a local-first retrieval system combining DuckDB, SciSpaCy, FAISS, and NetworkX, optimizing query-time memory usage by decoupling heavy UMLS linking from the inference path.

---

## Tech Stack
- **Data & Processing:** Python, Pandas, DuckDB, Parquet
- **NLP & Knowledge Graphs:** SciSpaCy, UMLS, NetworkX
- **Search & Embeddings:** SentenceTransformers, FAISS
- **Infrastructure:** AWS Open Data (boto3)
- **LLM:** DeepSeek API

---

## Future Ideas
- Set up a formal retrieval evaluation suite with curated biomedical queries.
- Improve lexical normalization for edge-case query phrasing.
- Package the inference step into a lightweight FastAPI service.

---

## Disclaimer
*This repository is a portfolio project focused on data engineering, NLP, and information retrieval. It is not a medical device, diagnostic system, or clinical decision support tool.*
