import pytest
import pandas as pd
import numpy as np
from fulcrum_app.intelligence_engine.primitives.descriptive_stats import calculate_descriptive_stats

def test_descriptive_stats_outlier_counts():
    # Construct a small dataset with an extreme outlier
    data = [10, 12, 15, 20, 21, 22, 1000]  # 1000 is an outlier
    df = pd.DataFrame({"value": data})
    stats = calculate_descriptive_stats(df, value_col="value")

    # We expect at least 1 outlier via z-score approach (1000 is far from mean)
    assert stats["outlier_count_z"] >= 1, f"Z-score outliers found: {stats['outlier_count_z']}"

    # With default iqr_multiplier=1.5, 1000 will definitely be out-of-bounds
    assert stats["outlier_count_iqr"] >= 1, f"IQR outliers found: {stats['outlier_count_iqr']}"

def test_descriptive_stats_coefficient_of_variation():
    # Data with moderate variability
    df = pd.DataFrame({"value": [50, 52, 48, 49, 51]})
    stats = calculate_descriptive_stats(df, value_col="value")
    assert stats["mean"] is not None
    assert stats["std"] is not None
    assert stats["cv"] == pytest.approx(stats["std"] / stats["mean"], 1e-7)

def test_descriptive_stats_zero_mean_for_cv():
    # If mean=0, CV is None
    df = pd.DataFrame({"value": [0, 0, 0]})
    stats = calculate_descriptive_stats(df, value_col="value")
    assert stats["mean"] == 0
    assert stats["cv"] is None

def test_calculate_required_growth():
    # normal scenario
    rate = calculate_required_growth(100, 200, 4, allow_negative=False)
    # we want (200/100)^(1/4) - 1 => 2^(1/4)-1 => ~0.189 => 18.9%
    assert rate is not None
    assert abs(rate - 0.189) < 0.01  # approximate

    # negative or zero
    assert calculate_required_growth(0, 100, 5) is None
    assert calculate_required_growth(-100, 50, 5) is None
    assert calculate_required_growth(100, 0, 5) is None

    # If allow_negative, let's do current_value=-100, target_value=-50 => less negative
    rate2 = calculate_required_growth(-100, -50, 2, allow_negative=True)
    # ratio = 50/100=0.5 => 0.5^(1/2)-1 => ~ -29.3% => negative growth needed => interesting domain case
    assert rate2 is not None and rate2 < 0

    # if periods_left=0 => None
    assert calculate_required_growth(100, 200, 0) is None
