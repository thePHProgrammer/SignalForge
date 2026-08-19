import pandas as pd
import pytest

from signalforge.backtest.walkforward import walk_forward_windows

D0 = pd.Timestamp("2024-01-01", tz="UTC")


def _day(n: int) -> pd.Timestamp:
    return D0 + pd.Timedelta(days=n)


def test_worked_example_default_step_matches_test_period():
    windows = list(
        walk_forward_windows(_day(0), _day(200), pd.Timedelta(days=60), pd.Timedelta(days=30))
    )
    assert len(windows) == 4
    assert windows[0].train_start == _day(0)
    assert windows[0].train_end == _day(60)
    assert windows[0].test_start == _day(60)
    assert windows[0].test_end == _day(90)
    assert windows[1].train_start == _day(30)
    assert windows[1].test_start == _day(90)
    assert windows[1].test_end == _day(120)
    assert windows[2].test_start == _day(120)
    assert windows[2].test_end == _day(150)
    assert windows[3].test_start == _day(150)
    assert windows[3].test_end == _day(180)
    # a 5th window's test_end (210) would exceed end (200) -- correctly rejected


def test_train_end_equals_test_start_always():
    windows = list(walk_forward_windows(_day(0), _day(200), pd.Timedelta(days=60), pd.Timedelta(days=30)))
    for w in windows:
        assert w.train_end == w.test_start


def test_step_equals_test_period_gives_contiguous_test_windows():
    windows = list(
        walk_forward_windows(_day(0), _day(150), pd.Timedelta(days=30), pd.Timedelta(days=30), step=pd.Timedelta(days=30))
    )
    for a, b in zip(windows, windows[1:]):
        assert a.test_end == b.test_start


def test_step_less_than_test_period_overlaps():
    windows = list(
        walk_forward_windows(_day(0), _day(150), pd.Timedelta(days=30), pd.Timedelta(days=30), step=pd.Timedelta(days=15))
    )
    assert len(windows) > 1
    for a, b in zip(windows, windows[1:]):
        assert b.test_start < a.test_end  # overlap


def test_step_greater_than_test_period_leaves_gaps():
    windows = list(
        walk_forward_windows(_day(0), _day(200), pd.Timedelta(days=30), pd.Timedelta(days=30), step=pd.Timedelta(days=45))
    )
    assert len(windows) > 1
    for a, b in zip(windows, windows[1:]):
        assert b.test_start > a.test_end  # gap


def test_range_shorter_than_one_window_yields_nothing():
    windows = list(walk_forward_windows(_day(0), _day(50), pd.Timedelta(days=60), pd.Timedelta(days=30)))
    assert windows == []


def test_no_truncated_trailing_window():
    # end at D185 -- the 4th window's test_end (D180) fits, but there's no
    # room for a 5th; the D180-D185 remainder must not appear as a partial window
    windows = list(walk_forward_windows(_day(0), _day(185), pd.Timedelta(days=60), pd.Timedelta(days=30)))
    assert len(windows) == 4
    assert windows[-1].test_end == _day(180)


@pytest.mark.parametrize("step", [pd.Timedelta(0), pd.Timedelta(days=-1)])
def test_non_positive_step_raises(step):
    with pytest.raises(ValueError):
        list(walk_forward_windows(_day(0), _day(200), pd.Timedelta(days=60), pd.Timedelta(days=30), step=step))


def test_non_positive_train_period_raises():
    with pytest.raises(ValueError):
        list(walk_forward_windows(_day(0), _day(200), pd.Timedelta(0), pd.Timedelta(days=30)))


def test_non_positive_test_period_raises():
    with pytest.raises(ValueError):
        list(walk_forward_windows(_day(0), _day(200), pd.Timedelta(days=60), pd.Timedelta(0)))
