from scoring.score_functions import ScoreFunction
import pandas as pd


def aggregate(
    df: pd.DataFrame,
    df_unscored: pd.DataFrame,
    agg_func,
    k: int,
    imputer,
    scorer: ScoreFunction,
    rank_column_name: str = "NRARank",
    complete_scores: bool = False,
    **kwargs
) -> dict:
    df = df.copy()
    cols = df.columns.tolist()
    score_column_mapping = scorer.get_mapping()
    unscored_cols = [score_column_mapping[c] for c in cols]
    n = len(df)

    # Precompute sorted index per column
    sorted_indices = {
        col: df[col].sort_values(ascending=False).index.tolist()
        for col in cols
    }
    min_per_col = df.min().to_dict()
    # Access tracking
    sorted_accesses = 0
    random_accesses = 0  # NRA does not use random access

    # Bookkeeping
    ptrs = {col: 0 for col in cols}
    seen_values = {}       # idx -> {col: value}
    lower_bounds = {}      # idx -> partial score
    upper_bounds = {}      # idx -> optimistic score
    best_seen = {}         # col -> best value seen so far

    done = False
    while not done:
        for col in cols:
            if ptrs[col] >= n:
                continue

            idx = sorted_indices[col][ptrs[col]]
            ptrs[col] += 1
            sorted_accesses += 1

            val = df.at[idx, col]
            best_seen[col] = val

            if idx not in seen_values:
                seen_values[idx] = {}

            seen_values[idx][col] = val

        # After each round, recompute bounds for all seen items
        for idx in seen_values:
            
            partial = pd.Series({**min_per_col, **seen_values[idx]})
            lower_bounds[idx] = agg_func(partial)
            
            # Upper bound with imputation
            raw_partial = df_unscored.loc[idx,unscored_cols].copy()
            raw_partial[unscored_cols] = [seen_values[idx].get(col, pd.NA) for col in cols]

            imputed_partial = imputer(raw_partial)
            imputed_scores = scorer.score(imputed_partial.to_frame().T).iloc[0].dropna()
            # scores_imputed = 
            # lower_bounds[idx] = agg_func(imputed_scores)

            # partial = pd.Series({**best_seen, **seen_values[idx]})
            upper_bounds[idx] = agg_func(imputed_scores)

        if len(lower_bounds) >= k:
            lower_bounds_series = pd.Series(lower_bounds)
            upper_bounds_series = pd.Series(upper_bounds)

            top_k = lower_bounds_series.sort_values(ascending=False).head(k)
            top_k_min_score = top_k.min()

            others = upper_bounds_series.drop(top_k.index, errors="ignore")
            if len(others) == 0 or (others <= top_k_min_score).all():
                done = True

    
    top_k_indices = pd.Series(lower_bounds).sort_values(ascending=False).head(k).index
    # Fully fetch all missing values from df for top-k indices
    if complete_scores:
        complete_scores_dict = {}
        for idx in top_k_indices:
            full_row = {
                col: seen_values[idx][col] if col in seen_values[idx] else df.at[idx, col]
                for col in cols
            }
            # Count as random access for missing columns
            random_accesses += sum(col not in seen_values[idx] for col in cols)
            complete_scores_dict[idx] = agg_func(pd.Series(full_row))

        top_k_indices = pd.Series(complete_scores_dict).sort_values(ascending=False).head(k).index


    return {
        "ranking": pd.Index(top_k_indices),
        "sorted_accesses": sorted_accesses,
        "random_accesses": random_accesses,
        "method": "nra"
    }