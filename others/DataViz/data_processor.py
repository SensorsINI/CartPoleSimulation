# data_processor.py

import pandas as pd
from tqdm import tqdm


class DataProcessor:
    """Computes absolute and relative errors, with outlier capping."""
    def __init__(self, config) -> None:
        self.config = config

    def compute_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.student_control_columns:
            return df

        t_cols = self.config.teacher_control_columns
        s_cols = self.config.student_control_columns
        pair_count = min(len(t_cols), len(s_cols))
        t_cols = t_cols[:pair_count]
        s_cols = s_cols[:pair_count]

        for t_col, s_col in tqdm(zip(t_cols, s_cols), desc='Computing Errors'):
            if t_col not in df.columns or s_col not in df.columns:
                continue
            abs_err = f"{s_col}_abs_err"
            df[abs_err] = (df[t_col] - df[s_col]).abs()

            rel_err = f"{s_col}_rel_err"
            eps = 1e-6
            df[rel_err] = df[abs_err] / (df[t_col].abs() + eps)
            df[rel_err] = df[rel_err].clip(upper=self.config.relative_error_cap)

            df[f"{s_col}_clipped"] = df[s_col].clip(-1, 1)

            abs_err_clipped = f"{s_col}_abs_err_clipped"
            df[abs_err_clipped] = (df[t_col] - df[f"{s_col}_clipped"]).abs()

            rel_err_clipped = f"{s_col}_rel_err_clipped"
            df[rel_err_clipped] = (
                df[abs_err_clipped] / (df[t_col].abs() + eps)
            ).clip(upper=self.config.relative_error_cap)

        return df
