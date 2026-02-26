# PMC_GraphRAG — Hybrid Graph-RAG for Citation-Grounded Biomedical Literature Retrieval

PMC_GraphRAG is a **biomedical literature retrieval system** built over the **PMC Open Access Subset**. Given a free-text symptom query, it retrieves **citation-backed evidence** from peer-reviewed articles by combining:

- **graph traversal** over extracted **Symptom → Condition → Evidence → Article** relationships, and
- **semantic similarity search** over chunk embeddings with **FAISS**.

The project is designed as an **applied NLP / retrieval systems pipeline**, not a diagnostic product. It does **not** use patient records, EHR data, or protected health information.

---

## Why this project exists

Naive RAG over medical text often returns vaguely relevant chunks without preserving **why** they matter or how they connect to a user’s symptoms. This project addresses that by building a structured evidence layer:

- symptoms and conditions are extracted from biomedical literature,
- mentions are linked to **UMLS concepts** offline,
- evidence chunks retain provenance back to source articles, and
- query-time retrieval combines **structured graph signals** with **semantic recall**.

The result is a retrieval pipeline that is more **inspectable**, **citation-safe**, and **explainable** than a vector-search-only chatbot.

---

## What the system does

### Included
- Ingests **PMC Open Access** articles with explicit **license filtering**
- Parses **JATS XML** into structured paragraph-level text
- Restricts retrieval candidates to **Results** and **Discussion** sections
- Builds retrieval-ready chunks with deterministic IDs
- Extracts **UMLS-grounded** symptom and condition mentions offline using **SciSpaCy**
- Constructs a provenance-preserving graph:
  - **Symptom ↔ Condition ↔ Evidence Chunk ↔ Article**
- Builds a local **FAISS** index for embedding-based retrieval
- Combines **graph traversal + semantic retrieval** at query time
- Optionally uses **DeepSeek** to compose final grounded responses from retrieved evidence

### Explicitly excluded
- No patient data
- No EHRs / MIMIC / protected health information
- No medical diagnosis or treatment recommendation
- No model training on clinical outcome labels

---

## Quantitative snapshot

From the current run:

- **1,000** PMC OA articles requested
- **994** unique PMCIDs parsed
- **99.4%** end-to-end parse success
- **88,995** paragraph-level rows extracted
- **20,838** Results/Discussion paragraph rows retained
- **12,935** retrieval-ready evidence chunks
- **5,779** UMLS-grounded concepts
- **609,882** provenance-linked evidence rows
- **290,910** aggregated symptom-condition edges
- **~0.23–0.25 sec** hybrid retrieval latency on the current sanity-query suite

These numbers reflect a **local-first**, notebook-driven build, not a hosted production service.

---

## Core idea

**Hybrid retrieval** in this project means:

1. **Graph path**
   - map query symptoms to known concepts
   - traverse **symptom → condition** edges
   - pull supporting evidence chunks and source articles

2. **Embedding path**
   - encode the query
   - retrieve semantically similar chunks with **FAISS**
   - map chunks back to candidate conditions via evidence tables

3. **Merge / rerank**
   - combine graph and embedding candidates
   - keep provenance and evidence diversity
   - optionally pass only retrieved evidence to the LLM

This makes the LLM a **constrained writer over retrieved evidence**, not the primary reasoning engine.

---

## Engineering Design & Trade-offs

During development, several architectural choices were made to prioritize system stability, memory management, and evidence quality over generic conversational abilities:

- **Offline High-Recall Graph:** The offline graph is intended to capture broad co-occurrence across the literature. Building a dense graph enables robust traversal, moving the computational burden of pruning to a runtime concern rather than an ingestion bottleneck. Explicit provenance is kept for every edge so citations are always inspectable.
- **Solving Query-Time Memory Constraints:** Attempting to load the full `UmlsEntityLinker` at inference caused a ~400MB memory spike on local hardware. To keep the query path lightweight, inference relies on NER + lexical grounding, mapping cheaply to the concepts already grounded in the offline graph.
- **Why Hybrid Retrieval is Necessary:** Relying solely on ontology/lexical grounding misses user phrasing variations. The FAISS semantic embedding search acts as a high-recall fallback path, recovering relevant evidence when semantic mapping is weak. 
- **Token-based vs. Paragraph Chunking:** Token-based chunking (~350 tokens with 50 overlap) was chosen over paragraph boundaries, as it provides far more consistency for embedding models when dealing with structurally dense biomedical articles.
- **Downweighting Non-Specific Symptoms:** Symptoms like *fever* appear across a massive cross-section of diseases, making them poor discriminators. Without downweighting or gating based on symptom specificity, fever-heavy literature biases the condition rankings.

---

## Data source

- **Dataset:** PMC Open Access Subset
- **Access:** AWS Open Data
- **Bucket:** `pmc-oa-opendata`
- **Download method:** `boto3` with anonymous `UNSIGNED` access
- **Downloaded artifacts:** XML + JSON only
- **License filters applied at query time:**
  - `cc0_license`, `cc_by_license`, `cc_by-sa_license`, `cc_by-nd_license`

---

## Repository structure

```text
PMC_GRAPHRAG/
  data/
    duckdb/
    index/
      chunks.faiss
      chunk_lookup.parquet
      chunk_meta.json
    parsed/
      graph_concepts.parquet
      graph_edge_evidence.parquet
      graph_edges_symptom_condition.parquet
      graph_edges.parquet
      graph_mentions.raw.parquet
      graph_mentions.parquet
      graph_nodes.parquet
      graph_nx.pkl
      pmc_raw_articles.parquet
      pmc_retrieval_candidates.parquet
      query_logs.jsonl
    raw/
  notebooks/
    01_download_parse.ipynb
    02_sql_filtering.ipynb
    03_scispacy_umls.ipynb
    04_hybrid_retrieval.ipynb
    05_query_compose.ipynb
  scripts/
  .env
```

---

## Pipeline overview

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

### 1) `01_download_parse.ipynb` — Ingestion + JATS XML parsing
Builds the raw corpus layer, kept strictly as a "plumbing-only" pipeline (ingestion + structural parse, no retrieval logic or NLP).

**What it does:**
- Queries PMC with a condition term (initially sepsis) plus license filters
- Downloads XML + JSON metadata from the PMC OA bucket
- Parses JATS XML using namespace-agnostic extraction
- Extracts: `pmcid`, `version`, `article_title`, `section_raw`, `paragraph_index`, `paragraph_text`, `license`
- Excludes retracted content using JSON metadata

**Output:**
- `data/parsed/pmc_raw_articles.parquet`

**Current ingestion metrics:**
- Requested articles: 1,000
- Unique PMCIDs parsed: 994
- Parse success rate: 99.4%
- Missing XML/JSON artifact directories: 0
- XML parse failures: 0

### 2) `02_sql_filtering.ipynb` — Evidence layer construction
Converts raw paragraphs into retrieval-ready evidence chunks, prioritizing evidence quality over maximizing volume.

**What it does:**
- Loads parsed data into DuckDB
- Keeps the latest version per PMCID
- Normalizes sections and retains Results + Discussion
- Builds chunks at approximately 350 tokens with 50-token overlap
- Drops short tail chunks
- Assigns stable deterministic `chunk_id`
- Exports a clean retrieval-candidate layer

**Output:**
- `data/parsed/pmc_retrieval_candidates.parquet`

**Current evidence-layer metrics:**
- Raw parsed rows: 88,995
- Filtered Results/Discussion rows: 20,838
- Retrieval chunks: 12,935
- Unique PMCIDs retained: 814
- Mean token count per chunk: ~333
- Mean chunks per article: ~15.9

### 3) `03_scispacy_umls.ipynb` — UMLS linking + graph artifact generation
Builds the offline graph layer.

**What it does:**
- Runs SciSpaCy over evidence chunks
- Links extracted entities to UMLS
- Filters linked mentions into:
  - Symptoms / findings
  - Conditions / diseases
- Creates:
  - raw mention table
  - deduplicated mention table
  - concept table
  - evidence triples
  - aggregated symptom-condition edges
  - graph export tables

**Outputs:**
- `graph_concepts.parquet`
- `graph_mentions.raw.parquet`
- `graph_mentions.parquet`
- `graph_edge_evidence.parquet`
- `graph_edges_symptom_condition.parquet`
- `graph_nodes.parquet`
- `graph_edges.parquet`
- `graph_nx.pkl`

**Current graph metrics:**
- UMLS-grounded concepts: 5,779
- Evidence rows: 609,882
- Symptom-condition edges: 290,910

### 4) `04_hybrid_retrieval.ipynb` — Embeddings + FAISS index
Builds the semantic retrieval layer.

**What it does:**
- Embeds each evidence chunk with SentenceTransformers
- Uses `all-MiniLM-L6-v2` as the baseline encoder
- Normalizes vectors for cosine-style retrieval
- Builds a FAISS `IndexFlatIP` index
- Persists lookup metadata for chunk reconstruction

**Outputs:**
- `data/index/chunks.faiss`
- `data/index/chunk_lookup.parquet`
- `data/index/chunk_meta.json`

**Current embedding/index metrics:**
- Embedded chunks: 12,935
- Embedding dimension: 384
- FAISS index size: 12,935
- FAISS artifact size: ~18.95 MB

### 5) `05_query_compose.ipynb` — Hybrid retrieval + evidence composition
Runs end-to-end inference, keeping retrieval deterministic and inspectable. 

**What it does:**
- Extracts symptom signals from the query
- Uses graph traversal when symptom grounding is strong enough
- Uses FAISS retrieval as a fallback / complement
- Merges graph and embedding candidates
- Builds a citation-ready evidence pack
- Optionally calls DeepSeek to produce a constrained final response

**Current query-time metrics:**
- Hybrid retrieval latency on sanity-query suite: ~0.23–0.25 sec
- Final evidence packs commonly include multiple distinct PMCIDs
- Retrieval remains functional without LLM generation

---

## Why this is different from a standard RAG project

Most RAG projects stop at:
- load documents
- chunk text
- embed chunks
- retrieve top-k
- send to LLM

This project goes further by adding:
- ontology-grounded entity extraction
- explicit symptom–condition graph construction
- offline/online separation for memory-safe inference
- evidence provenance attached to every retrieval hop
- hybrid structured + semantic retrieval
- citation-oriented output design

That makes it closer to an information retrieval / evidence systems project than a generic chatbot demo.

---

## Evaluation Notes (Retrieval Behavior)

This project focuses on **retrieval behavior diagnostics** rather than clinical accuracy benchmarks.

Key observed patterns from the current query sanity suite:

- **Graph vs Embedding Overlap:** Top-K condition overlap between graph traversal and FAISS retrieval varies by query (Jaccard@10 ≈ 0.0–0.6), indicating that the two paths provide complementary signals rather than redundant results.
- **Hybrid Recall Benefit:** Embedding retrieval consistently recovers relevant conditions when symptom grounding is weak or underspecified, improving recall beyond graph-only traversal.
- **Evidence Diversity:** Final evidence packs typically include **multiple distinct PMCIDs** (often 8–16 per query), reducing single-paper dominance and improving citation robustness.
- **Fallback Behavior:** When no specific symptoms are extracted (e.g., very generic queries), the system degrades gracefully to embedding-only retrieval rather than returning empty results.
- **Latency:** End-to-end hybrid retrieval completes in **~0.23–0.25 seconds** per query on local hardware, excluding optional LLM generation.

These diagnostics are intended to validate **system behavior and robustness**, not to claim medical predictive performance.

---

## How to run

### 1. Create an environment
Example with Conda:
```bash
conda create -n pmc-graphrag python=3.10
conda activate pmc-graphrag
```
Install dependencies:
```bash
pip install -r requirements.txt
```
If you use DeepSeek generation, also install:
```bash
pip install python-dotenv
```

### 2. Set environment variables
Create a `.env` file in the repo root:
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```
Then load it in notebooks that call DeepSeek:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Run notebooks in order
1. `01_download_parse.ipynb`
2. `02_sql_filtering.ipynb`
3. `03_scispacy_umls.ipynb`
4. `04_hybrid_retrieval.ipynb`
5. `05_query_compose.ipynb`

### 4. Scale the corpus
In `01_download_parse.ipynb`, adjust:
```python
MAX_ARTICLES = 1000
```
Then rerun downstream notebooks so artifacts stay consistent. At minimum, rerun Notebooks 02, 03, 04, and 05.

---

## Main artifacts

### Raw layer
- `pmc_raw_articles.parquet`: Paragraph-level parsed corpus with section structure and metadata.

### Evidence layer
- `pmc_retrieval_candidates.parquet`: Retrieval-ready evidence chunks with `chunk_id`, `pmcid`, `article_title`, `section`, `chunk_text`, and token metadata.

### Graph layer
- `graph_concepts.parquet` — extracted UMLS concepts
- `graph_mentions.raw.parquet` — raw mention-level output
- `graph_mentions.parquet` — deduplicated mention mapping
- `graph_edge_evidence.parquet` — symptom-condition evidence with chunk/article provenance
- `graph_edges_symptom_condition.parquet` — aggregated symptom-condition edges
- `graph_nodes.parquet`, `graph_edges.parquet` — exportable graph tables
- `graph_nx.pkl` — serialized NetworkX graph

### Retrieval layer
- `chunks.faiss` — FAISS index over chunk embeddings
- `chunk_lookup.parquet` — row-to-chunk lookup
- `chunk_meta.json` — retrieval model metadata

### Logging
- `query_logs.jsonl` — Saved query-time outputs for inspection and debugging.

---

## Example use case

A user enters symptoms such as: **fever, hypotension, confusion**

The system:
1. extracts symptom signals from the query,
2. traverses literature-derived symptom-condition relationships,
3. retrieves supporting evidence chunks from PMC OA articles,
4. merges graph and semantic retrieval results, and
5. returns possible literature-supported condition associations with citations.

*This is meant to support evidence discovery, not diagnosis.*

---

## Example Evidence Output (Illustrative)

**Input symptoms:** `fever, hypotension, confusion`

**Top literature-associated condition:** **Sepsis**

**Supporting evidence (excerpted):**

- **PMC1234567 — Results section** *“…patients presenting with hypotension and altered mental status were significantly more likely to develop septic shock…”*

- **PMC2345678 — Discussion section** *“…fever combined with circulatory instability remains a key clinical indicator associated with sepsis-related outcomes…”*

- **PMC3456789 — Results section** *“…confusion and low blood pressure were observed frequently among patients diagnosed with severe sepsis…”*

**Notes:**
- Evidence is retrieved directly from **peer-reviewed PMC Open Access articles**.
- Associations reflect **literature co-occurrence**, not diagnosis.
- Each snippet is traceable to its source article and section.

---

## Safety and scope

- This project does not diagnose.
- It is designed for literature-grounded association retrieval.
- It does not use private clinical data.
- Outputs should be interpreted as research evidence summaries.
- Real medical concerns should always be evaluated by a qualified clinician.

---

## Known limitations

- Query-time symptom grounding is intentionally lightweight, so some phrasing may not map cleanly.
- Single generic symptoms can still produce diffuse rankings.
- Graph edges are high-recall and require runtime pruning.
- Current evaluation is based on system behavior, evidence diversity, and retrieval sanity checks rather than a clinical gold-label benchmark.
- Scaling to larger corpora requires rerunning downstream graph and retrieval artifacts.

---

## Resume-ready highlights

- Built a hybrid Graph-RAG biomedical retrieval system over 1,000 PMC Open Access articles, achieving 99.4% parse success and producing 12.9K evidence chunks for downstream retrieval.
- Constructed an explainable graph of 5.8K UMLS-grounded concepts, 609K provenance-linked evidence rows, and 290K symptom-condition edges connecting symptoms, conditions, chunks, and source articles.
- Designed a local-first retrieval pipeline combining DuckDB, SciSpaCy, SentenceTransformers, FAISS, and NetworkX, with lightweight query-time inference and optional DeepSeek response composition.

---

## Tech stack

- **Languages/Tools:** Python, Pandas, DuckDB
- **NLP & Graph:** SciSpaCy, UMLS linking, NetworkX
- **Embeddings & Search:** SentenceTransformers, FAISS
- **Infrastructure:** boto3, Parquet
- **LLM:** DeepSeek API (optional generation)

---

## Future improvements

- Add a formal retrieval evaluation suite over curated symptom queries.
- Improve aliasing / lexical normalization for query-time symptom grounding.
- Expand beyond an initial single-condition search seed.
- Add visualization for graph neighborhoods and evidence provenance.
- Package query-time retrieval into a lightweight API or interface.

---

## Acknowledgments

- PubMed Central Open Access Subset
- NCBI / PMC
- AWS Open Data
- SciSpaCy
- UMLS
- SentenceTransformers
- FAISS

---

## Disclaimer

This repository is a research / portfolio project focused on biomedical literature retrieval and citation-grounded evidence synthesis. It is not a medical device, diagnostic system, or clinical decision support tool.