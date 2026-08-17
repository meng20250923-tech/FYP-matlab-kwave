"""Trigonometric interpolation translated from ``triginterp.m``."""

from __future__ import annotations

import numpy as np


def trigcardinal(x: np.ndarray, n: int) -> np.ndarray:
    """Evaluate the cardinal basis of an ``n``-point trigonometric interpolant.

    Args:
        x: Scaled distances from one interpolation node.
        n: Number of equispaced interpolation nodes.

    Returns:
        Cardinal basis values with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if n % 2:
            values = np.sin(n * np.pi * x / 2) / (n * np.sin(np.pi * x / 2))
        else:
            values = np.sin(n * np.pi * x / 2) / (n * np.tan(np.pi * x / 2))
    return np.where(x == 0, 1.0, values)


def triginterp(xi: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the trigonometric interpolant on equispaced nodes."""
    xi = np.asarray(xi, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    n = x.size
    if n < 2:
        raise ValueError("At least two interpolation nodes are required.")
    scale = (x[1] - x[0]) / (2.0 / n)
    x, xi = x / scale, xi / scale
    return sum(y[k] * trigcardinal(xi - x[k], n) for k in range(n))
