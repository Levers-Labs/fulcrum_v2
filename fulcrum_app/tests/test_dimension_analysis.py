def test_calculate_slice_metrics_top_n():
    # Construct df with a high-card dimension
    data = {
        "category": ["A","B","C","D","E","F"],
        "value": [100, 50, 50, 20, 10, 5]
    }
    df = pd.DataFrame(data)
    result = calculate_slice_metrics(df, "category", "value", agg="sum", top_n=3)
    # Expect A,B,C in the top 3, plus an 'Other' row for D/E/F
    assert len(result) == 4
    assert "Other" in result["category"].values

def test_compute_slice_shares():
    agg_df = pd.DataFrame({
        "category": ["A", "B", "C"],
        "aggregated_value": [50, 30, 20]
    })
    result = compute_slice_shares(agg_df, "category", val_col="aggregated_value")
    # total=100 => shares=50%,30%,20%
    assert abs(result.loc[result["category"]=="A","share_pct"].iloc[0] - 50.0) < 1e-9

def test_detect_anomalies_in_slices():
    # small dataset with repeated slices
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=6),
        "segment": ["X","X","X","Y","Y","Z"],
        "value": [10, 12, 11, 100, 110, 500]  # Z has only 1 data point
    })
    out = detect_anomalies_in_slices(df, "segment", "value", date_col="date")
    # only latest row of each segment is tested => 
    # segment X's latest= row 2 => ~ 11 vs mean(10,12,11)=11, std=~1 => no anomaly
    # segment Y's latest= row 4 => ~110 vs mean(100,110)=105 std ~7 => check if 110 is > 3*7 => probably not
    # segment Z's latest= row 5 => but slice_count=1 => no anomaly
    # So likely all false
    assert not out["slice_anomaly"].any()

def test_concentration_index_hhi():
    df = pd.DataFrame({"aggregated_value":[50,30,20]})
    hhi = calculate_concentration_index(df, method="HHI")
    # shares= .5,.3,.2 => sum of squares= .25 + .09 + .04= .38
    assert abs(hhi-0.38)<1e-9

def test_concentration_index_gini():
    df = pd.DataFrame({"aggregated_value":[50,30,20]})
    gini_val = calculate_concentration_index(df, method="gini")
    # Expect some number >0
    assert gini_val>0