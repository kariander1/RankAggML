import pandas as pd
import numpy as np

class ScoreFunction:
    def __init__(self, df: pd.DataFrame, normalize=False, columns_to_drop = [], **kwargs):
        """
        Initialize and precompute everything needed for scoring.
        """
        self.raw_transform_map = {
            'Price': lambda x: -x,
            'Comfort': lambda x: x,
            'Beauty': lambda x: x,
            'R1': lambda x: x,
            'R2': lambda x: x,
            'R3': lambda x: x,
            'num_platforms': lambda x: -x,
        }

        self.grouped_norm = {
            "Sales": ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
        }

        self.score_column_mapping = {}
        self.normalization_params = {}
        self.global_sales_max = None
        self.min_year = None
        self.normalize = normalize
        self.columns_to_drop = columns_to_drop
        self._precompute(df)

    def _precompute(self, df: pd.DataFrame):
        df = df.copy()

        # Raw transforms
        for col, transform in self.raw_transform_map.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not col in self.columns_to_drop:
                transformed = transform(df[col])
                min_val = transformed.min()
                max_val = transformed.max()
                self.normalization_params[col] = (min_val, max_val)
                self.score_column_mapping[f'Score_{col}'] = col

        # Earliest year normalization (exponential)
        if 'earliest_year' in df.columns and pd.api.types.is_numeric_dtype(df['earliest_year']) and not 'earliest_year' in self.columns_to_drop:
            self.min_year = df['earliest_year'].min()
            self.score_column_mapping['Score_year'] = 'earliest_year'

        # Grouped sales normalization
        all_sales = self.grouped_norm["Sales"]
        valid_sales = [col for col in all_sales if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if valid_sales:
            self.global_sales_max = df[valid_sales].values.max()
            for col in valid_sales:
                if not col in self.columns_to_drop:
                    self.score_column_mapping[f'Score_{col}'] = col

    def _normalize_series(self, series, min_val, max_val, lower=0, upper=1):
        if min_val == max_val:
            return pd.Series([upper] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val) * (upper - lower) + lower

    def _normalize_exponential(self, series, alpha=0.1):
        shifted = series - self.min_year
        scores = 1 - np.exp(-alpha * shifted)
        return scores / scores.max()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        score_data = {}
        score_order = []

        # Raw transforms
        for col in df.columns:
            if col in self.normalization_params:
                min_val, max_val = self.normalization_params[col]
                transformed = self.raw_transform_map[col](df[col])
                score_col = f'Score_{col}'
                score_order.append(score_col)
                if self.normalize:
                    score_data[score_col] = self._normalize_series(transformed, min_val, max_val)
                else:
                    score_data[score_col] = transformed

        # Year
        if self.min_year is not None and 'earliest_year' in df.columns:
            score_col = 'Score_year'
            score_order.append(score_col)
            score_data[score_col] = self._normalize_exponential(pd.to_numeric(df['earliest_year'], errors='coerce'))

        # Grouped sales
        if self.global_sales_max is not None:
            for col in df.columns:
                if col in self.grouped_norm.get("Sales", []):
                    score_col = f'Score_{col}'
                    score_order.append(score_col)
                    score_data[score_col] = df[col] / self.global_sales_max

        # Reorder columns based on score_order
        score_df = pd.DataFrame(score_data, index=df.index)
        return score_df[score_order]

    def get_mapping(self) -> dict:
        return self.score_column_mapping.copy()
