import pandas as pd

def aggregate(
    df: pd.DataFrame,
    agg_func,
    k: int,
    rank_column_name: str = "NaiveRank",
    **kwargs
) -> dict:
    df = df.copy()
    rank_df = pd.DataFrame(index=df.index)

    # Rank each column
    for col in df.columns:
        rank_df[col + "_rank"] = df[col].rank(ascending=False, method='average')

    # Aggregate the ranks
    df[rank_column_name] = rank_df.apply(agg_func, axis=1)

    # Get top-k indices based on aggregated rank
    ranking = df[rank_column_name].sort_values(ascending=False).head(k).index

    return {
        "ranking": ranking,
        "sorted_accesses": 0,
        "random_accesses": len(df) * len(df.columns),  # full access to all values
        "method": "naive"
    }
