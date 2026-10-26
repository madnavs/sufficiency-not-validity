"""Exact paired McNemar test used for every comparison in the paper."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact binomial McNemar. a, b are paired 0/1 outcomes."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0:
        return n10, n01, 1.0
    p = stats.binomtest(n10, n01 + n10, 0.5).pvalue
    return n10, n01, p
