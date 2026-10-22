"""Consumer-facing period summaries must follow session order, not row order."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ensemble.backtest import summarize_returns


def test_period_drawdowns_are_invariant_to_input_row_order():
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6),
            "horizon": 1,
            "strategy": "ensemble",
            "return": [1.0, -0.5, -0.5, 1.0, -0.5, -0.5],
            "turnover": 0.0,
        }
    )
    holdout_start = str(returns.iloc[3]["date"].date())
    chronological = summarize_returns(returns, holdout_start)
    shuffled = summarize_returns(returns.iloc[[1, 0, 2, 4, 3, 5]], holdout_start)
    assert_frame_equal(chronological, shuffled)

    # Each subperiod grows from 1 to 2, then declines to 0.5: a 75% drawdown.
    # The full path ends at 0.25 after the first peak of 2: an 87.5% drawdown.
    drawdowns = shuffled.set_index("period")["max_drawdown"]
    assert drawdowns["development"] == pytest.approx(-0.75)
    assert drawdowns["holdout"] == pytest.approx(-0.75)
    assert drawdowns["all"] == pytest.approx(-0.875)
