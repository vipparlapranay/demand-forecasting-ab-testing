import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from demand_forecasting.data import generate_demand
from demand_forecasting.experiments import analyze_ab_test, generate_ab_experiment
from demand_forecasting.forecast import BacktestConfig, rolling_origin_backtest


def run(
    output_dir: Path,
    input_csv: Path | None = None,
    metrics_path: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_csv is None:
        demand = generate_demand()
        demand.to_csv(output_dir / "synthetic_weekly_demand.csv", index=False)
    else:
        demand = pd.read_csv(input_csv, parse_dates=["week_start"])

    forecast_metrics, predictions = rolling_origin_backtest(
        demand,
        BacktestConfig(),
    )
    predictions.to_csv(output_dir / "forecast_backtest_predictions.csv", index=False)

    experiment = generate_ab_experiment()
    experiment_metrics = analyze_ab_test(experiment)
    experiment.to_csv(output_dir / "synthetic_ab_experiment.csv", index=False)
    results = {
        "forecasting": forecast_metrics,
        "ab_test": experiment_metrics,
        "disclaimer": (
            "Metrics are from reproducible synthetic data; they are not "
            "production or business-impact results."
        ),
    }
    serialized = json.dumps(results, indent=2) + "\n"
    (output_dir / "metrics.json").write_text(serialized, encoding="utf-8")
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(serialized, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest weekly demand and analyze a powered A/B test."
    )
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("results/metrics.json"),
        help="Save a reproducible benchmark summary outside the generated-output folder.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = run(args.output_dir, args.input_csv, args.metrics_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
