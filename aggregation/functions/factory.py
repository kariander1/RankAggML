import numpy as np

def get_agg_function(name: str, **kwargs):
    name = name.lower()
    if name == "avg":
        return lambda row: row.mean()
    elif name == "sum":
        return lambda row: row.sum()
    elif name == "min":
        return lambda row: row.min()
    elif name == "max":
        return lambda row: row.max()
    elif name == "geometric":
        return lambda row: np.prod(row) ** (1 / len(row))
    elif name == "harmonic":
        return lambda row: len(row) / np.sum(1.0 / row)
    elif name == "median":
        return lambda row: row.median()
    elif name == "softmax":
        alpha = kwargs.get("alpha", 1.0)
        return lambda row: np.mean(np.exp(alpha * row))
    elif name.startswith("lp-"):
        p = float(name.split("-")[1])
        return lambda row: (np.sum(row**p)) ** (1/p)
    elif name.startswith("borda"):
        m = kwargs["n_items"]   # total number of items (rows)
        ranks_df = kwargs["ranks"]  # full rank matrix (same shape as scores)
        return lambda row: np.sum([max(0, m - (ranks_df.at[row.name, col] )) for col in row.index])
    else:
        raise ValueError(f"Unknown aggregation function: {name}")
