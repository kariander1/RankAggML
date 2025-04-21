import pandas as pd
from pathlib import Path


def get_dataset(name: str,combine_sales=False) -> pd.DataFrame:
    """
    Factory method to return a DataFrame based on dataset name.

    Args:
        name (str): Dataset name. Can be one of:
            - "toy"
            - "toy2"
            - "vgsales"
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data_root = Path("data")

    if name == "toy":
        return pd.read_csv(data_root / "toy.csv")

    elif name == "toy2":
        return pd.read_csv(data_root / "toy2.csv")

    elif name == "vgsales":
        df = pd.read_csv(data_root / "vgsales.csv")
        if not combine_sales:
            return df
        # Clean strings to handle potential inconsistencies
        df["Name"] = df["Name"].str.strip()
        df["Genre"] = df["Genre"].astype(str).str.strip()
        df["Publisher"] = df["Publisher"].astype(str).str.strip()

        # Group by game name (ignore platform)
        aggregated = df.groupby("Name").agg(
            earliest_year=("Year", "min"),
            platforms=("Platform", lambda x: " | ".join(sorted(set(x)))),
            num_platforms=("Platform", lambda x: len(set(x))),
            genres=("Genre", lambda x: " | ".join(sorted(set(x)))),
            publishers=("Publisher", lambda x: " | ".join(sorted(set(x)))),
            NA_Sales=("NA_Sales", "sum"),
            EU_Sales=("EU_Sales", "sum"),
            JP_Sales=("JP_Sales", "sum"),
            Other_Sales=("Other_Sales", "sum"),
            Global_Sales=("Global_Sales", "sum"),
        ).reset_index()

        return aggregated

    else:
        raise ValueError(f"Unknown dataset name: {name}")
