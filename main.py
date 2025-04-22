import pandas as pd
import pyrallis
import numpy as np

from pathlib import Path
from config import Config
from scoring.score_functions import ScoreFunction
from aggregation.factory import get_aggregator
from aggregation.functions.factory import get_agg_function
from imputation.factory import get_imputer
from data.factory import get_dataset
from data.erasure import erase_values
from utils.metrics import compute_accuracy_metrics
import random

def save_config(cfg: Config, save_path: str):
    
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
    pyrallis.dump(cfg, open(save_path,'w'))
    print(f"\n[✔] Config saved to: {save_path}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
@pyrallis.wrap()
def main(cfg: Config):
    save_path = Path(cfg.output_path) / "config.yaml"
    save_config(cfg, save_path)
    set_seed(cfg.seed)

    df_raw = get_dataset(cfg.dataset_name, **cfg.dataset_kwargs)
    
    df = erase_values(df_raw, cfg.p_erase)

    if cfg.p_erase > 0:
        print(f"\n[!] Erased {cfg.p_erase * 100:.2f}% of the dataset")
        print(f"[!] Number of erased values: {df.isna().sum().sum()}")
        print(f"[!] Number of rows with erased values: {df.isna().any(axis=1).sum()}")
        df.to_csv(Path(cfg.output_path) / "dataset_erased.csv", index=False)

    imputer = get_imputer(cfg.imputer_name, df=df, **cfg.imputer_kwargs)
    df = imputer(df)

    if cfg.p_erase > 0:
        df.to_csv(Path(cfg.output_path) / "dataset_imputed.csv", index=False)

    scorer_raw = ScoreFunction(df, **cfg.scorer_kwargs)
    scores_raw = scorer_raw.score(df_raw)
    
    scorer = ScoreFunction(df, **cfg.scorer_kwargs)
    scores = scorer.score(df)

    scores.to_csv(Path(cfg.output_path) / "scores.csv", index=False)
    borda_ranks = scores.rank(axis=0, ascending=False, method="min")

    # Handle list or single k
    k_values = cfg.k if isinstance(cfg.k, (list, tuple)) else [cfg.k]

    # Store cumulative rows
    metrics_list = []
    results_list = []

    for k_val in k_values:
        if len(df) < k_val:
            print(f"\n[!] k={k_val} is larger than the number of items in the dataset. Skipping this k.")
            continue
        output_k_dir = Path(cfg.output_path) / f'k_{k_val}'
        output_k_dir.mkdir(parents=True, exist_ok=True)

        aggregator = get_aggregator(cfg.aggregator_name)
        naive_aggregator = get_aggregator("naive")
        agg_func = get_agg_function(cfg.agg_function_name, n_items=len(df), ranks=borda_ranks)

        result_dict = aggregator(df=scores, agg_func=agg_func, k=k_val, imputer=imputer, df_unscored=df, scorer=scorer, **cfg.aggregator_kwargs)
        naive_result_dict = naive_aggregator(df=scores_raw, agg_func=agg_func, k=k_val, **cfg.aggregator_kwargs)

        # Accuracy metrics
        metrics = compute_accuracy_metrics(naive_result_dict['ranking'], result_dict['ranking'])
        print(f"\n[✔] Accuracy metrics for k={k_val}:")
        print(metrics)
        metrics["k"] = k_val
        metrics_list.append(metrics)

        # Top-k
        top_k_df = df.loc[result_dict["ranking"]]
        top_k_df.to_csv(output_k_dir / "top_k.csv", index=False)
        print(f"\n[k={k_val}] Top-k Ranked Items:")
        print(top_k_df)

        # Access stats
        print(f"\nMethod: {result_dict['method']}")
        print(f"Sorted accesses: {result_dict['sorted_accesses']}")
        print(f"Random accesses: {result_dict['random_accesses']}")
        print(f"Total accesses: {result_dict['sorted_accesses'] + result_dict['random_accesses']}")

        result_row = {
            "k": k_val,
            "sorted_accesses": result_dict["sorted_accesses"],
            "random_accesses": result_dict["random_accesses"],
            "total_accesses": result_dict["sorted_accesses"] + result_dict["random_accesses"]
        }
        results_list.append(result_row)

    # Save combined metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(Path(cfg.output_path) / "metrics.csv", index=False)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(Path(cfg.output_path) / "results.csv", index=False)

if __name__ == "__main__":
    main()
