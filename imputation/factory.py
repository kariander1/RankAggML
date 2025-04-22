# imputation/factory.py
from .heuristic_imputer import HeuristicImputer
from .ml_imputer import MLSalesImputer
from .basic_imputer import BasicImputer
def get_imputer(name: str, **kwargs) -> callable:
    if name == "drop":
        return drop
    elif name == "basic":
        return BasicImputer(**kwargs).impute
    elif name == "heuristic":
        return HeuristicImputer(**kwargs).impute
    elif name == "ml":
        return MLSalesImputer(**kwargs).impute
    else:
        raise ValueError(f"Unknown imputer name: {name}")

def drop(df, **kwargs):
    """
    Drop rows with missing values in the specified columns.
    """
    return df.dropna()