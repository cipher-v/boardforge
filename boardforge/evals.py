from __future__ import annotations

from dataclasses import dataclass
from boardforge.models import Match


def precision_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    top = predicted[:k]
    return 0.0 if not top else sum(x in relevant for x in top) / len(top)


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return 0.0 if not relevant else len(set(predicted[:k]) & relevant) / len(relevant)


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    import math
    dcg = sum((1.0 if x in relevant else 0.0) / math.log2(i + 2) for i, x in enumerate(predicted[:k]))
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def hard_violation_rate(matches: list[Match]) -> float:
    return 0.0 if not matches else sum(bool(m.hard_violations) for m in matches) / len(matches)


def explanation_coverage(matches: list[Match]) -> float:
    return 0.0 if not matches else sum(bool(m.reasons) for m in matches) / len(matches)


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb)
