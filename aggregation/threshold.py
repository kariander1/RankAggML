import pandas as pd
import heapq

def aggregate(df: pd.DataFrame, agg_func, k: int, rank_column_name: str = "ThresholdRank", **kwargs) -> dict:
    df = df.copy()
    cols = df.columns.tolist()
    n = len(df)

    sorted_indices = {
        col: df[col].sort_values(ascending=False).index.tolist()
        for col in cols
    }

    seen_scores = {}
    top_k_heap = []
    last_seen_values = {col: None for col in cols}
    ptrs = {col: 0 for col in cols}

    sorted_accesses = 0
    random_accesses = 0

    done = False
    while not done:
        for col in cols:
            if ptrs[col] >= n:
                continue

            idx = sorted_indices[col][ptrs[col]]
            ptrs[col] += 1
            sorted_accesses += 1

            last_seen_values[col] = df.at[idx, col]

            if idx not in seen_scores:
                score = agg_func(df.loc[idx])
                seen_scores[idx] = score
                random_accesses += len(cols)

                heapq.heappush(top_k_heap, (score, idx))
                if len(top_k_heap) > k:
                    heapq.heappop(top_k_heap)

            if all(val is not None for val in last_seen_values.values()):
                threshold = agg_func(pd.Series(last_seen_values))
                if len(top_k_heap) >= k and all(score >= threshold for score, _ in top_k_heap):
                    done = True
                    break

    top_k_indices = [idx for _, idx in sorted(top_k_heap, reverse=True)]

    return {
        "ranking": pd.Index(top_k_indices),
        "sorted_accesses": sorted_accesses,
        "random_accesses": random_accesses,
        "method": "threshold"
    }
