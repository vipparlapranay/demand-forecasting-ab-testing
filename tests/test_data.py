import pandas as pd
import pytest

from demand_forecasting.data import generate_demand, load_or_generate


def test_generator_is_reproducible_with_expected_series_volume() -> None:
    first = generate_demand(weeks=60, products=2, stores=3, seed=7)
    second = generate_demand(weeks=60, products=2, stores=3, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 60 * 2 * 3
    assert first["demand"].ge(0).all()


def test_csv_loader_rejects_duplicate_series_week(tmp_path) -> None:
    path = tmp_path / "demand.csv"
    pd.DataFrame(
        {
            "week_start": ["2025-01-06", "2025-01-06"],
            "product_id": ["SKU-1", "SKU-1"],
            "store_id": ["STORE-1", "STORE-1"],
            "demand": [1, 2],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="one row per week/product/store"):
        load_or_generate(path)
