import numpy as np
import pandas as pd
import pytest

from demand_forecasting.data import generate_demand
from demand_forecasting.forecast import BacktestConfig, rolling_origin_backtest, wape


def test_wape_aggregates_error_and_guards_zero_denominator() -> None:
    assert wape(np.array([100.0, 100.0]), np.array([90.0, 110.0])) == 10.0
    with pytest.raises(ValueError, match="zero"):
        wape(np.array([0.0]), np.array([0.0]))


def test_rolling_origin_backtest_has_complete_nonnegative_predictions() -> None:
    data = generate_demand(weeks=120, products=2, stores=2, seed=23)
    config = BacktestConfig(
        horizon_weeks=2,
        folds=2,
        fold_step_weeks=4,
        min_train_weeks=104,
    )
    metrics, predictions = rolling_origin_backtest(data, config)

    assert metrics["rolling_origin_folds"] == 2
    assert metrics["holdout_weeks_scored"] == 4
    assert metrics["holdout_series_week_observations"] == 16
    assert predictions["model_prediction"].ge(0).all()
    assert predictions["seasonal_naive_prediction"].ge(0).all()
    assert np.isfinite(metrics["lightgbm_wape_pct"])
