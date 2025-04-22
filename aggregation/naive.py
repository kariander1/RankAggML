import pandas as pd

def aggregate(
    df: pd.DataFrame,
    agg_func,
    k: int,
    rank_column_name: str = "NaiveRank",
    **kwargs
) -> dict:
    df = df.copy()
    ranks = df.apply(agg_func, axis=1)
    # Aggregate the ranks
    # df[rank_column_name] = rank_df.apply(agg_func, axis=1)

    # Get top-k indices based on aggregated rank
    ranking = ranks.sort_values(ascending=False).head(k).index

    return {
        "ranking": ranking,
        "sorted_accesses": 0,
        "random_accesses": len(df) * len(df.columns),  # full access to all values
        "method": "naive"
    }
