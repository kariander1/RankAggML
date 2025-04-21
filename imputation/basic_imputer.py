import pandas as pd

def basic_impute(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generic basic imputation:
    - Numeric columns: fill with mean
    - Object/string columns: fill with mode, or 'No Data' if mode not available
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric: fill with mean
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # Categorical/string: fill with mode or "No Data"
                try:
                    mode_value = df[col].mode()[0]
                except IndexError:
                    mode_value = "No Data"
                df[col].fillna(mode_value, inplace=True)

    return df
