from tqdm import tqdm
tqdm.pandas()

import pandas as pd
import numpy as np

class HeuristicImputer:
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize the imputer with a reference DataFrame.
        Precomputes statistics and ratios for imputation logic.
        """
        self.df = df.copy()

        self.sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        self._compute_sales_ratios()

    def _compute_sales_ratios(self):
        """Precompute average sales ratios for missing value estimation."""
        existing = self.df.dropna(subset=self.sales_cols)
        if existing.empty:
            self.sales_ratios = {col: 0.25 for col in self.sales_cols}
        else:
            row_sums = existing[self.sales_cols].sum(axis=1)
            self.sales_ratios = {
                col: (existing[col] / row_sums).mean()
                for col in self.sales_cols
            }

    def _impute_earliest_year(self, row):
        """Estimate earliest year based on similar publisher + platform titles."""
        if pd.notnull(row.get("earliest_year")):
            return row["earliest_year"]

        candidates = self.df[
            (self.df['publishers'] == row.get('publishers')) &
            (self.df['platforms'].notna()) &
            (self.df['earliest_year'].notnull())
        ]

        # Match if platform overlaps
        row_platforms = set(str(row.get("platforms", "")).split(" | "))
        candidates = candidates[
            candidates["platforms"].apply(
                lambda ps: bool(row_platforms & set(str(ps).split(" | ")))
            )
        ]

        if not candidates.empty:
            return round(candidates["earliest_year"].mean())
        return np.nan  # fallback to later mean fill

    def _impute_sales(self, row):
        """Estimate missing sales values using sales ratios."""
        present = {col: row[col] for col in self.sales_cols if pd.notnull(row[col])}
        missing = [col for col in self.sales_cols if pd.isnull(row[col])]

        if not present:
            return row  # can't impute anything

        known_total = sum(present.values())
        known_ratio = sum(self.sales_ratios[col] for col in present)
        estimated_total = known_total / known_ratio if known_ratio > 0 else 0

        for col in missing:
            row[col] = estimated_total * self.sales_ratios[col]

        return row

    def _impute_num_platforms(self, row):
        """Fill missing num_platforms from platform string."""
        if pd.notnull(row.get("num_platforms")):
            return row["num_platforms"]

        platforms = row.get("platforms")
        if pd.notnull(platforms):
            return len(str(platforms).split(" | "))
        return np.nan  # fallback handled later

    def impute(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run the heuristic imputation on the provided DataFrame.
        If df is None, imputes the original training data used in init.
        """
        df = df.copy() if df is not None else self.df.copy()
        if isinstance(df, pd.Series):
            return self.impute(df.to_frame().T).iloc[0]
        # Heuristic rules
        if "earliest_year" in df.columns:
            df["earliest_year"] = df.progress_apply(self._impute_earliest_year, axis=1)

        df = df.progress_apply(self._impute_sales, axis=1)

        if "num_platforms" in df.columns:
            df["num_platforms"] = df.progress_apply(self._impute_num_platforms, axis=1)

        # Fallback: mean/mode for remaining missing
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    try:
                        mode_value = df[col].mode()[0]
                    except IndexError:
                        mode_value = "No Data"
                    df[col].fillna(mode_value, inplace=True)

        return df
