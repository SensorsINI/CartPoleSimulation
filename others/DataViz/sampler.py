# sampler.py

import pandas as pd


class Sampler:
    """Samples states in various ways."""
    def __init__(self, config) -> None:
        self.config = config

    def sample_high_error(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        if not self.config.student_control_columns:
            return pd.DataFrame()
        abs_err_columns = [c for c in df.columns if c.endswith("_abs_err")]
        if not abs_err_columns:
            return pd.DataFrame()
        df_copy = df.copy()
        df_copy["total_abs_err"] = df_copy[abs_err_columns].sum(axis=1)
        return df_copy.sort_values("total_abs_err", ascending=False).head(top_n)

    def sample_poorly_represented_regions(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()
        return df.sample(n=min(n, len(df)), replace=False)

    def sample_uniform(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame()
        return df.sample(n=min(n, len(df)), replace=False)
