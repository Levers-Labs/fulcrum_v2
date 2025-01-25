import pandas as pd
from fulcrum_app.intelligence_engine.primitives.time_series import calculate_pop_growth

def test_calculate_pop_growth():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4, freq="D"),
        "value": [100, 110, 90, 95]
    })
    df_sorted = df.sort_values("date")
    result = calculate_pop_growth(df_sorted)

    # growth from 100->110 is 10%
    assert result.loc[1, "pop_growth"] == 10.0
    # growth from 110->90 is ~ -18.18%
    assert abs(result.loc[2, "pop_growth"] + 18.1818) < 0.001
