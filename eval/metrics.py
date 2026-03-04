from __future__ import annotations
from typing import List, Set
import math

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    topk = set(retrieved[:k])
    return 1.0 if len(topk.intersection(relevant)) > 0 else 0.0

def mrr(retrieved: List[str], relevant: Set[str], k: int = 50) -> float:
    if not relevant:
        return float("nan")
    for i, docid in enumerate(retrieved[:k], start=1):
        if docid in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    def dcg(items: List[str]) -> float:
        s = 0.0
        for idx, d in enumerate(items, start=1):
            rel = 1.0 if d in relevant else 0.0
            s += rel / math.log2(idx + 1)
        return s

    if not relevant:
        return float("nan")

    actual = dcg(retrieved[:k])
    # binary ideal upper bound
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return actual / ideal if ideal > 0 else 0.0
