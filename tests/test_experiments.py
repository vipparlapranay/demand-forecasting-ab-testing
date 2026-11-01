import pandas as pd
import pytest

from demand_forecasting.experiments import (
    analyze_ab_test,
    generate_ab_experiment,
    required_sample_size,
)


def test_power_analysis_returns_positive_per_arm_sample_size() -> None:
    sample_size = required_sample_size()
    assert sample_size > 0
    assert required_sample_size(minimum_detectable_effect=0.01) > sample_size


def test_ab_analysis_is_reproducible_and_reports_two_sided_test() -> None:
    first = generate_ab_experiment(seed=12)
    second = generate_ab_experiment(seed=12)
    pd.testing.assert_frame_equal(first, second)

    result = analyze_ab_test(first)
    assert result["test"] == "two-sided two-proportion z-test"
    assert result["control_users"] == 10_000
    assert result["treatment_users"] == 10_000
    assert 0 <= result["p_value"] <= 1
    assert result["required_users_per_group_for_mde"] == required_sample_size()


def test_ab_analysis_rejects_repeated_experiment_units() -> None:
    experiment = generate_ab_experiment(users_per_group=4)
    experiment.loc[1, "user_id"] = experiment.loc[0, "user_id"]
    with pytest.raises(ValueError, match="exactly once"):
        analyze_ab_test(experiment)
