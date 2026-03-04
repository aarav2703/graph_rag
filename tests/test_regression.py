from __future__ import annotations

import json
from pathlib import Path
import pytest

from pmc_graphrag import GraphRAGPipeline
from eval.metrics import recall_at_k

GOLDEN = Path("tests/golden.jsonl")

@pytest.fixture(scope="session")
def pipe():
    return GraphRAGPipeline(project_root=Path(".").resolve())

def _load():
    rows = []
    with GOLDEN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def test_hybrid_runs(pipe):
    res = pipe.retrieve("fever, low blood pressure, confusion", mode="hybrid", top_conditions=5)
    retrieved = pipe.retrieved_pmcids_from_context(res.context, max_docs=30)
    assert res.latency_sec >= 0.0
    # Don't assert retrieved contains the placeholder PMCIDs yet.

def test_golden_smoke(pipe):
    rows = _load()
    for r in rows:
        res = pipe.retrieve(r["query"], mode="hybrid", top_conditions=5)
        retrieved = pipe.retrieved_pmcids_from_context(res.context, max_docs=50)
        rel = set(map(str, r.get("relevant_pmcids", [])))
        _ = recall_at_k(retrieved, rel, 20)
        assert res.latency_sec >= 0.0
