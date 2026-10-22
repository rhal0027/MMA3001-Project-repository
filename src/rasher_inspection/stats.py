"""Small statistics helpers for reporting results from small samples."""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Better than the usual ``p ± z*sqrt(p(1-p)/n)`` for small ``n`` or for
    proportions near 0 or 1, which is the situation for most counts in this
    project (e.g. 3 missed defects in 12 reviewed images).

    Args:
        successes: Number of "events" (e.g. images with a missed defect).
        n: Sample size.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(lower, upper)`` bounds of the proportion, in [0, 1].

    Raises:
        ValueError: If ``n`` is not positive or ``successes`` is out of range.

    Example:
        >>> lo, hi = wilson_interval(3, 12)
        >>> round(lo, 2), round(hi, 2)
        (0.09, 0.53)
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
