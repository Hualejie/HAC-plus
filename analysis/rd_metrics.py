"""Rate-distortion metrics used by the Phase 2D aggregation."""

import math

import numpy as np


def _prepare(points):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("RD points must be (rate, quality) pairs")
    if values.shape[0] < 4:
        raise ValueError("BD-rate requires at least four RD points")
    if np.any(values[:, 0] <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("RD rates must be positive and all values finite")
    order = np.argsort(values[:, 1], kind="stable")
    values = values[order]
    if np.any(np.diff(values[:, 1]) <= 0):
        raise ValueError("BD-rate quality values must be strictly increasing")
    return values


def bd_rate(control_points, test_points):
    """Return the cubic Bjontegaard delta-rate of test against control.

    Each input contains ``(rate, quality)`` pairs. A negative percentage means
    the test curve uses fewer bytes at equal quality.
    """

    control = _prepare(control_points)
    test = _prepare(test_points)
    quality_low = max(control[0, 1], test[0, 1])
    quality_high = min(control[-1, 1], test[-1, 1])
    if quality_high <= quality_low:
        raise ValueError("RD curves have no overlapping quality interval")

    control_poly = np.polyfit(control[:, 1], np.log(control[:, 0]), 3)
    test_poly = np.polyfit(test[:, 1], np.log(test[:, 0]), 3)
    control_integral = np.polyint(control_poly)
    test_integral = np.polyint(test_poly)
    span = quality_high - quality_low
    average_log_difference = (
        np.polyval(test_integral, quality_high)
        - np.polyval(test_integral, quality_low)
        - np.polyval(control_integral, quality_high)
        + np.polyval(control_integral, quality_low)
    ) / span
    return {
        "bd_rate_percent": (math.exp(float(average_log_difference)) - 1.0) * 100.0,
        "quality_low": float(quality_low),
        "quality_high": float(quality_high),
        "control_points": int(control.shape[0]),
        "test_points": int(test.shape[0]),
    }
