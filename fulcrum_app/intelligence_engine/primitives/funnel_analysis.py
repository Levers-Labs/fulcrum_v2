import pandas as pd
from typing import List, Dict

def perform_funnel_analysis(df: pd.DataFrame, steps: List[str], step_col: str="funnel_step", value_col: str="value"):
    """
    Generic funnel calculation:
      - Expects df with columns [step_col, value_col], 
        e.g. step_col indicates 'Signup', 'Activated', 'Upgraded'...
      - steps is an ordered list: ["Signup","Activated","Upgraded"]
    Returns a list of dictionaries describing each funnel step's count, 
      conversion rate from previous, etc.

    This is a minimal placeholder. 
    In real usage, you might pass different sets of user IDs over time.
    """
    # Basic approach: for each step in steps, sum the value where step_col == step
    funnel_data = []
    prev_count = None
    for i, s in enumerate(steps):
        step_df = df[df[step_col] == s]
        count_s = step_df[value_col].sum()
        conv_rate = None
        if prev_count is not None and prev_count != 0:
            conv_rate = count_s / prev_count
        funnel_data.append({"step": s, "count": count_s, "conversion_rate": conv_rate})
        prev_count = count_s
    return funnel_data
