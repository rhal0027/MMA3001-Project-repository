"""Timing helpers for inference speed.

A production line passes a fixed number of trays per minute, which sets a
time budget per image. These helpers measure inference time repeatably:
warm-up runs are discarded (the first calls include model loading, memory
allocation and GPU kernel compilation), and the measurement is repeated so a
spread can be reported, not just one number.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable, Sequence


def time_per_item(fn: Callable, items: Sequence, warmup: int = 5,
                  repeats: int = 3) -> dict:
    """Time ``fn(item)`` for each item, repeated over the whole sequence.

    Args:
        fn: Function to time, called once per item. Its result is ignored.
        items: Inputs (e.g. image paths).
        warmup: Number of untimed calls first, using the first items.
        repeats: Number of timed passes over all items.

    Returns:
        ``{"n_items", "repeats", "mean_ms", "std_ms", "median_ms", "p95_ms",
        "min_ms", "max_ms"}`` over all timed calls, in milliseconds per item.

    Raises:
        ValueError: If ``items`` is empty or ``repeats`` < 1.
    """
    if not items or repeats < 1:
        raise ValueError("need at least one item and one repeat")
    for item in list(items)[:warmup]:
        fn(item)
    times = []
    for _ in range(repeats):
        for item in items:
            t0 = time.perf_counter()
            fn(item)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "n_items": len(items),
        "repeats": repeats,
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def trays_per_minute(ms_per_image: float) -> float:
    """Convert a per-image time to the maximum trays per minute it allows.

    Example:
        >>> trays_per_minute(20.0)
        3000.0
    """
    return 60_000.0 / ms_per_image


def environment() -> dict:
    """Describe the hardware the timings were measured on.

    Timings are only meaningful with this context: Colab assigns different
    GPUs between sessions.

    Returns:
        Python version, CPU description, and GPU name if PyTorch sees one.
    """
    info = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["gpu"] = None
    return info
