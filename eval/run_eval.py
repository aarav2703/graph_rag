from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import argparse
import pandas as pd

from eval.metrics import recall_at_k, mrr, ndcg_at_k
from pmc_graphrag import GraphRAGPipeline


def load_queries(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=str, default="eval/queries_silver.jsonl")
    ap.add_argument("--out", type=str, default="eval/results.csv")
    ap.add_argument("--mode", type=str, choices=["graph", "vector", "hybrid", "all"], default="all")
    ap.add_argument("--top_conditions", type=int, default=5)
    ap.add_argument("--max_docs", type=int, default=50)
    args = ap.parse_args()

    pipe = GraphRAGPipeline(project_root=Path(".").resolve())

    queries = load_queries(Path(args.queries))
    modes = ["graph", "vector", "hybrid"] if args.mode == "all" else [args.mode]

    out_rows = []
    for q in queries:
        qid = q.get("qid", "")
        text = q["query"]
        relevant = set(map(str, q.get("relevant_pmcids", [])))

        for mode in modes:
            res = pipe.retrieve(text, mode=mode, top_conditions=args.top_conditions)
            retrieved = pipe.retrieved_pmcids_from_context(res.context, max_docs=args.max_docs)

            out_rows.append({
                "qid": qid,
                "mode": mode,
                "latency_sec": res.latency_sec,
                "retrieved_count": len(retrieved),
                "recall@5": recall_at_k(retrieved, relevant, 5),
                "recall@10": recall_at_k(retrieved, relevant, 10),
                "recall@20": recall_at_k(retrieved, relevant, 20),
                "mrr@50": mrr(retrieved, relevant, 50),
                "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
            })

    df = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = df.groupby("mode").agg(
        n=("qid", "count"),
        p50_latency=("latency_sec", lambda x: float(x.quantile(0.50))),
        p95_latency=("latency_sec", lambda x: float(x.quantile(0.95))),
        recall10=("recall@10", "mean"),
        recall20=("recall@20", "mean"),
        mrr50=("mrr@50", "mean"),
        ndcg10=("ndcg@10", "mean"),
    ).reset_index()

    print(summary.to_string(index=False))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
