import pandas as pd

class BasicImputer:

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.reference_df = df.copy()

    def impute(self, data):
        if isinstance(data, pd.Series):
            return self._impute_dataframe(pd.DataFrame(data).T).iloc[0]
        elif isinstance(data, pd.DataFrame):
            return self._impute_dataframe(data.copy())
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")

    def _impute_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col not in self.reference_df.columns:
                raise ValueError(f"Column '{col}' not found in reference DataFrame.")

            if df[col].isnull().any():
                ref_col = self.reference_df[col]
                if pd.api.types.is_numeric_dtype(ref_col):
                    df[col].fillna(ref_col.mean(), inplace=True)
                else:
                    try:
                        mode_value = ref_col.mode()[0]
                    except IndexError:
                        mode_value = "No Data"
                    df[col].fillna(mode_value, inplace=True)

        return df
