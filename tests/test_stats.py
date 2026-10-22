import pytest

from rasher_inspection.stats import wilson_interval


def test_interval_contains_point_estimate():
    lo, hi = wilson_interval(3, 12)
    assert lo < 3 / 12 < hi


def test_zero_successes_gives_nonzero_upper_bound():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2


@pytest.mark.parametrize("k,n", [(1, 0), (-1, 5), (6, 5)])
def test_invalid_inputs(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)
