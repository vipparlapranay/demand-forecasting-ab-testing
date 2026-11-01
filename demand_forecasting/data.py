from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "week_start"
GROUP_COLUMNS = ["product_id", "store_id"]
TARGET_COLUMN = "demand"


def generate_demand(
    weeks: int = 208,
    products: int = 12,
    stores: int = 8,
    seed: int = 20261101,
) -> pd.DataFrame:
    if weeks < 60 or products < 1 or stores < 1:
        raise ValueError("Use at least 60 weeks and one product and store.")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
    product_ids = [f"SKU-{number:03d}" for number in range(1, products + 1)]
    store_ids = [f"STORE-{number:02d}" for number in range(1, stores + 1)]
    product_levels = rng.uniform(16, 105, size=products)
    store_factors = rng.uniform(0.55, 1.65, size=stores)
    rows = []

    for product_number, product_id in enumerate(product_ids):
        for store_number, store_id in enumerate(store_ids):
            level = product_levels[product_number] * store_factors[store_number]
            phase = rng.uniform(0, 2 * np.pi)
            series_noise = rng.uniform(0.09, 0.20)
            previous = level
            for index, week_start in enumerate(dates):
                annual = 1 + 0.18 * np.sin(
                    2 * np.pi * index / 52.18 + phase
                )
                secondary = 1 + 0.05 * np.cos(
                    4 * np.pi * index / 52.18 + phase / 2
                )
                trend = 1 + 0.00055 * index
                promo = int(rng.random() < 0.055)
                expected = level * annual * secondary * trend * (
                    1 + promo * 0.28
                )
                demand = max(
                    0,
                    int(
                        round(
                            0.48 * previous
                            + 0.52 * expected
                            + rng.normal(0, expected * series_noise)
                        )
                    ),
                )
                previous = demand
                rows.append(
                    {
                        DATE_COLUMN: week_start,
                        "product_id": product_id,
                        "store_id": store_id,
                        TARGET_COLUMN: demand,
                        "promo_flag": promo,
                    }
                )

    return pd.DataFrame(rows).sort_values(
        [DATE_COLUMN, *GROUP_COLUMNS], ignore_index=True
    )


def load_or_generate(
    path: Path | None,
    weeks: int = 208,
    products: int = 12,
    stores: int = 8,
    seed: int = 20261101,
) -> pd.DataFrame:
    if path is None:
        return generate_demand(weeks, products, stores, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Demand CSV not found: {path}")
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    required = {DATE_COLUMN, *GROUP_COLUMNS, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Demand CSV is missing columns: {', '.join(sorted(missing))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Date, product, store, and demand values cannot be null.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError("Demand values cannot be negative.")
    if frame.duplicated([DATE_COLUMN, *GROUP_COLUMNS]).any():
        raise ValueError("Demand CSV must have one row per week/product/store.")
    return frame.sort_values([DATE_COLUMN, *GROUP_COLUMNS], ignore_index=True)
