"""Port of Subvox/Decoding/DTW.swift: dynamic time warping with a Sakoe-Chiba band."""

from __future__ import annotations

import math


def _euclidean(x: list[float], y: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(x, y)))


def distance(a: list[list[float]], b: list[list[float]], band_fraction: float = 0.25) -> float:
    """Normalised DTW distance; lets the same mouthed word match when said faster or slower."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return math.inf
    band = max(abs(n - m), int(max(n, m) * band_fraction))
    inf = math.inf
    previous = [inf] * (m + 1)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = [inf] * (m + 1)
        lo = max(1, i - band)
        hi = min(m, i + band)
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            cost = _euclidean(ai, b[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    # Normalise by the maximum path length so long and short words compare fairly.
    return previous[m] / (n + m)
