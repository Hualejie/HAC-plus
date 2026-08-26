import pytest

from analysis.rd_metrics import bd_rate


def test_bd_rate_identical_curves_are_zero():
    points = [(100, 25), (200, 27), (400, 29), (800, 31)]
    assert bd_rate(points, points)["bd_rate_percent"] == pytest.approx(0.0)


def test_bd_rate_recovers_constant_rate_reduction():
    control = [(100, 25), (200, 27), (400, 29), (800, 31)]
    test = [(90, 25), (180, 27), (360, 29), (720, 31)]
    assert bd_rate(control, test)["bd_rate_percent"] == pytest.approx(-10.0)


def test_bd_rate_rejects_non_overlapping_quality_ranges():
    control = [(100, 20), (200, 21), (400, 22), (800, 23)]
    test = [(100, 24), (200, 25), (400, 26), (800, 27)]
    with pytest.raises(ValueError, match="no overlapping"):
        bd_rate(control, test)
