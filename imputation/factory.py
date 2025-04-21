# imputation/factory.py
from .heuristic_imputer import HeuristicImputer
from .ml_imputer import MLSalesImputer
from .basic_imputer import basic_impute
def get_imputer(name: str, **kwargs) -> callable:
    if name == "basic":
        return basic_impute
    elif name == "heuristic":
        return HeuristicImputer(**kwargs).impute
    elif name == "ml":
        return MLSalesImputer(**kwargs).impute
    else:
        raise ValueError(f"Unknown imputer name: {name}")
