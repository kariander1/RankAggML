import pandas as pd

def aggregate(df: pd.DataFrame, agg_func, k: int, rank_column_name: str = "FaginRank", **kwargs) -> dict:
    df = df.copy()
    cols = df.columns.tolist()
    n = len(df)

    # Precompute sorted index per column
    sorted_indices = {col: df[col].sort_values(ascending=False).index.tolist() for col in cols}

    # Access tracking
    sorted_accesses = 0
    random_accesses = 0

    # Bookkeeping
    seen = {}  # idx -> set of columns where seen
    ptrs = {col: 0 for col in cols}

    fully_seen = set()
    while len(fully_seen) < k:
        for col in cols:
            if ptrs[col] >= n:
                continue  # end of list

            idx = sorted_indices[col][ptrs[col]]
            ptrs[col] += 1
            sorted_accesses += 1

            if idx not in seen:
                seen[idx] = set()
            seen[idx].add(col)

            if len(seen[idx]) == len(cols):
                fully_seen.add(idx)

            if len(fully_seen) >= k:
                break

    # Random access for all seen items (not just fully seen)
    all_seen_indices = pd.Index(seen.keys())
    seen_df = df.loc[all_seen_indices]
    random_accesses += len(seen_df) * len(cols)

    # Aggregate scores and pick top-k
    aggregated_scores = seen_df.apply(agg_func, axis=1)
    top_k_indices = aggregated_scores.sort_values(ascending=False).head(k).index

    return {
        "ranking": top_k_indices,
        "sorted_accesses": sorted_accesses,
        "random_accesses": random_accesses,
        "method": "fagin"
    }
