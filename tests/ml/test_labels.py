import pandas as pd
import pytest

from signalforge.ml.labels import label_forward_direction


def test_regression_label_reflects_actual_entry_not_a_close_reference():
    # The originally-buggy design compared close[T] to close[T+horizon] --
    # price could "rise" from that reference while the actual trade
    # (entered at open[T+1], which may have already gapped past the rise)
    # lost money. open[1]=110 is the actual entry, open[2]=105 is the actual
    # exit -- a real loss, must be labeled 0.0.
    open_ = pd.Series([100.0, 110.0, 105.0])
    label = label_forward_direction(open_, horizon=1)
    assert label.iloc[0] == 0.0


def test_hand_computed_horizon_1():
    open_ = pd.Series([100.0, 105.0, 110.0, 108.0, 120.0, 115.0])
    label = label_forward_direction(open_, horizon=1)
    assert label.iloc[0] == 1.0  # entry=105, exit=110 -> up
    assert label.iloc[1] == 0.0  # entry=110, exit=108 -> down
    assert label.iloc[2] == 1.0  # entry=108, exit=120 -> up
    assert label.iloc[3] == 0.0  # entry=120, exit=115 -> down
    assert label.iloc[4:].isna().all()


def test_hand_computed_horizon_2():
    open_ = pd.Series([100.0, 105.0, 110.0, 108.0, 120.0, 115.0])
    label = label_forward_direction(open_, horizon=2)
    assert label.iloc[0] == 1.0  # entry=105, exit=108 -> up
    assert label.iloc[1] == 1.0  # entry=110, exit=120 -> up
    assert label.iloc[2] == 1.0  # entry=108, exit=115 -> up
    assert label.iloc[3:].isna().all()


def test_horizon_must_be_positive():
    open_ = pd.Series([100.0, 101.0])
    with pytest.raises(ValueError):
        label_forward_direction(open_, horizon=0)
    with pytest.raises(ValueError):
        label_forward_direction(open_, horizon=-1)


def test_purge_worked_example():
    n, horizon = 20, 3
    open_local = pd.Series([100.0 + i for i in range(n)])  # rows 0..19

    label = label_forward_direction(open_local, horizon)

    # last (horizon+1)=4 rows purged: indices 16,17,18,19
    assert label.iloc[16:].isna().all()
    assert label.iloc[:16].notna().all()
    assert int(label.notna().sum()) == 16

    # row 15's label reads exit at row 15+horizon+1=19, the last row inside
    # the window: entry=open[16]=116, exit=open[19]=119 -> up
    assert label.iloc[15] == 1.0


def test_purge_is_leak_proof_against_data_just_past_the_window():
    n, horizon = 20, 3
    base_opens = [100.0 + i for i in range(n)]
    label_local = label_forward_direction(pd.Series(base_opens), horizon)

    # A distinct, wildly different price placed just past the local window's
    # end. If this function were ever accidentally called on a wider series
    # (e.g. one that reached into the test window), previously-purged rows
    # would suddenly pick up that future information.
    extended_opens = base_opens + [999_999.0]
    label_extended = label_forward_direction(pd.Series(extended_opens), horizon)

    # rows that were valid within the local window are unaffected by data
    # beyond it, purged or not
    pd.testing.assert_series_equal(label_local.iloc[:16], label_extended.iloc[:16], check_names=False)

    # row 16 was correctly purged (NaN) using only the local window's data --
    # but given one more row of (spiked) data, it flips to a real value
    # driven entirely by that spike. This is exactly the leak purging
    # prevents, demonstrated directly: never hand the function that extra
    # data in the first place (see ml/runner.py for how the train-local
    # slice is constructed).
    assert pd.isna(label_local.iloc[16])
    assert label_extended.iloc[16] == 1.0
