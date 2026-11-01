import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


def required_sample_size(
    baseline_rate: float = 0.12,
    minimum_detectable_effect: float = 0.015,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between zero and one.")
    treatment_rate = baseline_rate + minimum_detectable_effect
    if not 0 < treatment_rate < 1:
        raise ValueError("The detectable treatment rate must be between zero and one.")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between zero and one.")
    effect_size = abs(proportion_effectsize(baseline_rate, treatment_rate))
    per_group = NormalIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return math.ceil(float(per_group))


def generate_ab_experiment(
    users_per_group: int = 10_000,
    control_rate: float = 0.12,
    treatment_rate: float = 0.125,
    seed: int = 20261102,
) -> pd.DataFrame:
    if users_per_group < 2:
        raise ValueError("users_per_group must be at least two.")
    if not 0 <= control_rate <= 1 or not 0 <= treatment_rate <= 1:
        raise ValueError("Conversion rates must be between zero and one.")
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, control_rate, size=users_per_group)
    treatment = rng.binomial(1, treatment_rate, size=users_per_group)
    return pd.DataFrame(
        {
            "user_id": [
                f"CONTROL-{number:06d}" for number in range(users_per_group)
            ]
            + [f"TREATMENT-{number:06d}" for number in range(users_per_group)],
            "group": ["control"] * users_per_group
            + ["treatment"] * users_per_group,
            "converted": np.concatenate([control, treatment]),
        }
    )


def analyze_ab_test(
    experiment: pd.DataFrame,
    baseline_rate: float = 0.12,
    minimum_detectable_effect: float = 0.015,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> dict:
    required_columns = {"user_id", "group", "converted"}
    missing = required_columns - set(experiment.columns)
    if missing:
        raise ValueError(f"Experiment is missing columns: {', '.join(sorted(missing))}")
    if experiment[list(required_columns)].isna().any().any():
        raise ValueError("Experiment assignment and outcome cannot be null.")
    if experiment["user_id"].duplicated().any():
        raise ValueError("Each experiment unit must appear exactly once.")
    if not set(experiment["group"].unique()) == {"control", "treatment"}:
        raise ValueError("Experiment requires control and treatment groups.")
    if not experiment["converted"].isin([0, 1]).all():
        raise ValueError("Conversion outcomes must be binary values (0 or 1).")

    control = experiment.loc[experiment["group"] == "control", "converted"]
    treatment = experiment.loc[experiment["group"] == "treatment", "converted"]
    control_n, treatment_n = len(control), len(treatment)
    if min(control_n, treatment_n) < 2:
        raise ValueError("Each group must contain at least two units.")
    control_rate_observed = float(control.mean())
    treatment_rate_observed = float(treatment.mean())
    difference = treatment_rate_observed - control_rate_observed
    test_statistic, p_value = proportions_ztest(
        count=np.array([int(treatment.sum()), int(control.sum())]),
        nobs=np.array([treatment_n, control_n]),
        alternative="two-sided",
    )
    standard_error = math.sqrt(
        treatment_rate_observed * (1 - treatment_rate_observed) / treatment_n
        + control_rate_observed * (1 - control_rate_observed) / control_n
    )
    critical = float(norm.ppf(1 - alpha / 2))
    confidence_interval = (
        difference - critical * standard_error,
        difference + critical * standard_error,
    )
    return {
        "test": "two-sided two-proportion z-test",
        "alpha": alpha,
        "confidence_level_pct": (1 - alpha) * 100,
        "control_users": control_n,
        "treatment_users": treatment_n,
        "control_conversions": int(control.sum()),
        "treatment_conversions": int(treatment.sum()),
        "control_conversion_rate_pct": round(control_rate_observed * 100, 4),
        "treatment_conversion_rate_pct": round(treatment_rate_observed * 100, 4),
        "absolute_lift_percentage_points": round(difference * 100, 4),
        "relative_lift_pct": (
            round(difference / control_rate_observed * 100, 4)
            if control_rate_observed
            else None
        ),
        "lift_confidence_interval_percentage_points": [
            round(confidence_interval[0] * 100, 4),
            round(confidence_interval[1] * 100, 4),
        ],
        "z_statistic": round(float(test_statistic), 4),
        "p_value": round(float(p_value), 6),
        "statistically_significant": bool(p_value < alpha),
        "required_users_per_group_for_mde": required_sample_size(
            baseline_rate,
            minimum_detectable_effect,
            alpha,
            target_power,
        ),
        "planned_minimum_detectable_effect_percentage_points": (
            minimum_detectable_effect * 100
        ),
        "target_power_pct": target_power * 100,
        "conclusion": (
            "Significant conversion-rate difference at the pre-specified confidence level."
            if p_value < alpha
            else "No statistically significant conversion-rate lift at the 95% confidence level."
            if alpha == 0.05
            else "No statistically significant conversion-rate difference at the pre-specified confidence level."
        ),
    }
