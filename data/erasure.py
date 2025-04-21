import pandas as pd
def erase_values(df: pd.DataFrame, p:float) -> pd.DataFrame:
    """Erase values in the DataFrame to simulate missing data."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            # Erase numeric values with a probability p
            mask = df.sample(frac=p).index
            df.loc[mask, col] = pd.NA
    return df