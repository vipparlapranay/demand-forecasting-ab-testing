from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from demand_forecasting.data import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    TARGET_COLUMN,
)
from demand_forecasting.features import (
    FEATURE_COLUMNS,
    make_features,
    model_matrix,
)


@dataclass(frozen=True)
class BacktestConfig:
    horizon_weeks: int = 4
    folds: int = 5
    fold_step_weeks: int = 8
    min_train_weeks: int = 104
    seed: int = 2026


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    if denominator == 0:
        raise ValueError("WAPE is undefined when total actual demand is zero.")
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def _new_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=240,
        learning_rate=0.045,
        num_leaves=23,
        max_depth=-1,
        min_child_samples=24,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def rolling_origin_backtest(
    data: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[dict, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data[DATE_COLUMN].unique())))
    required = config.min_train_weeks + config.fold_step_weeks * (
        config.folds - 1
    ) + config.horizon_weeks
    if len(dates) < required:
        raise ValueError(
            f"Need at least {required} distinct weeks for the configured backtest; "
            f"got {len(dates)}."
        )
    if config.folds < 1 or config.horizon_weeks < 1:
        raise ValueError("Backtest folds and horizon must be positive.")
    if data.duplicated([DATE_COLUMN, *GROUP_COLUMNS]).any():
        raise ValueError("Input must have one row per week/product/store.")
    if (data[TARGET_COLUMN] < 0).any():
        raise ValueError("Demand cannot be negative.")

    first_origin = len(dates) - (
        config.fold_step_weeks * (config.folds - 1)
        + config.horizon_weeks
    )
    predictions: list[pd.DataFrame] = []
    feature_columns = list(FEATURE_COLUMNS)

    for fold_number in range(config.folds):
        origin_index = first_origin + fold_number * config.fold_step_weeks
        origin = dates[origin_index]
        history = data[pd.to_datetime(data[DATE_COLUMN]) < origin].copy()
        history_weeks = pd.DatetimeIndex(
            sorted(pd.to_datetime(history[DATE_COLUMN].unique()))
        )
        if len(history_weeks) < config.min_train_weeks:
            raise ValueError("A backtest fold has fewer than min_train_weeks.")

        training_features = make_features(history)
        training_features = training_features.dropna(subset=feature_columns)
        model = _new_model(config.seed + fold_number)
        model.fit(
            model_matrix(training_features),
            training_features[TARGET_COLUMN].astype(float),
        )

        simulated = history.copy()
        for step in range(config.horizon_weeks):
            target_date = dates[origin_index + step]
            actual_rows = data[pd.to_datetime(data[DATE_COLUMN]) == target_date]
            future = actual_rows.drop(columns=[TARGET_COLUMN]).copy()
            future[TARGET_COLUMN] = np.nan
            future["promo_flag"] = 0
            simulated = pd.concat([simulated, future], ignore_index=True)
            all_features = make_features(simulated)
            current = all_features[
                pd.to_datetime(all_features[DATE_COLUMN]) == target_date
            ].copy()
            estimate = np.maximum(
                0,
                model.predict(
                    model_matrix(current),
                    num_iteration=model.best_iteration_,
                ),
            )
            current["model_prediction"] = estimate
            current["actual"] = actual_rows[TARGET_COLUMN].to_numpy(dtype=float)

            seasonal_date = target_date - pd.Timedelta(weeks=52)
            seasonal = history[
                pd.to_datetime(history[DATE_COLUMN]) == seasonal_date
            ][[*GROUP_COLUMNS, TARGET_COLUMN]].rename(
                columns={TARGET_COLUMN: "seasonal_naive_prediction"}
            )
            current = current.merge(seasonal, on=GROUP_COLUMNS, how="left")
            if current["seasonal_naive_prediction"].isna().any():
                raise ValueError(
                    "Seasonal-naive baseline requires 52 weeks of prior observations."
                )
            current["seasonal_naive_prediction"] = current[
                "seasonal_naive_prediction"
            ].clip(lower=0)
            current["fold"] = fold_number + 1
            predictions.append(
                current[
                    [
                        DATE_COLUMN,
                        *GROUP_COLUMNS,
                        "actual",
                        "model_prediction",
                        "seasonal_naive_prediction",
                        "fold",
                    ]
                ]
            )
            simulated.loc[
                pd.to_datetime(simulated[DATE_COLUMN]) == target_date,
                TARGET_COLUMN,
            ] = estimate

    result = pd.concat(predictions, ignore_index=True)
    metrics = {
        "frequency": "weekly",
        "products": int(data["product_id"].nunique()),
        "stores": int(data["store_id"].nunique()),
        "product_store_series": int(
            data.groupby(GROUP_COLUMNS, observed=True).ngroups
        ),
        "history_weeks": int(len(dates)),
        "rolling_origin_folds": config.folds,
        "horizon_weeks_per_fold": config.horizon_weeks,
        "holdout_weeks_scored": int(
            result[DATE_COLUMN].nunique()
        ),
        "holdout_series_week_observations": int(len(result)),
        "lightgbm_wape_pct": round(
            wape(
                result["actual"].to_numpy(),
                result["model_prediction"].to_numpy(),
            ),
            4,
        ),
        "seasonal_naive_wape_pct": round(
            wape(
                result["actual"].to_numpy(),
                result["seasonal_naive_prediction"].to_numpy(),
            ),
            4,
        ),
    }
    metrics["wape_improvement_pct"] = round(
        (
            1
            - metrics["lightgbm_wape_pct"]
            / metrics["seasonal_naive_wape_pct"]
        )
        * 100,
        4,
    )
    metrics["beats_seasonal_naive"] = (
        metrics["lightgbm_wape_pct"] < metrics["seasonal_naive_wape_pct"]
    )
    return metrics, result
