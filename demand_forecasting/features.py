import numpy as np
import pandas as pd

from demand_forecasting.data import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    TARGET_COLUMN,
)

LAGS = (1, 2, 4, 8, 13, 26, 52)
ROLLING_WINDOWS = (4, 13, 26, 52)
FEATURE_COLUMNS = [
    *GROUP_COLUMNS,
    "week_of_year",
    "month",
    "quarter",
    "trend_index",
    "week_sin",
    "week_cos",
    "promo_flag",
    *(f"lag_{lag}" for lag in LAGS),
    *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
    *(f"rolling_std_{window}" for window in ROLLING_WINDOWS),
]


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    dates = pd.to_datetime(frame[DATE_COLUMN])
    iso_week = dates.dt.isocalendar().week.astype(int).clip(upper=52)
    frame["week_of_year"] = iso_week
    frame["month"] = dates.dt.month
    frame["quarter"] = dates.dt.quarter
    frame["trend_index"] = (
        (dates - dates.min()).dt.days // 7
    ).astype(int)
    frame["week_sin"] = np.sin(2 * np.pi * iso_week / 52.18)
    frame["week_cos"] = np.cos(2 * np.pi * iso_week / 52.18)
    if "promo_flag" not in frame.columns:
        frame["promo_flag"] = 0
    frame["promo_flag"] = frame["promo_flag"].fillna(0).astype(int)

    grouped = frame.groupby(GROUP_COLUMNS, sort=False)[TARGET_COLUMN]
    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = grouped.shift(1)
        frame[f"rolling_mean_{window}"] = shifted.groupby(
            [frame[column] for column in GROUP_COLUMNS], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=window).mean())
        frame[f"rolling_std_{window}"] = shifted.groupby(
            [frame[column] for column in GROUP_COLUMNS], sort=False
        ).transform(lambda values: values.rolling(window, min_periods=window).std())
    return frame


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[FEATURE_COLUMNS].copy()
    for column in GROUP_COLUMNS:
        matrix[column] = matrix[column].astype("category")
    return matrix
