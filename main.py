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
    
    df = get_dataset(cfg.dataset_name, **cfg.dataset_kwargs)

    df = erase_values(df, cfg.p_erase)

    imputer = get_imputer(cfg.imputer_name, df=df, **cfg.imputer_kwargs)
    
    df = imputer(df)

    scorer = ScoreFunction(df, **cfg.scorer_kwargs)
    scores = scorer.score(df)
    
    # Compute ranks for Borda count (if selected as agg function)
    borda_ranks = scores.rank(axis=0, ascending=False, method="min")

    # Select the aggregation algorithm (Fagin, NRA, Threshold, etc.)
    aggregator = get_aggregator(cfg.aggregator_name)
    
    # Select how to aggregate (sum/max/etc.)
    agg_func = get_agg_function(cfg.agg_function_name, n_items=len(df), ranks=borda_ranks)

    result_dict = aggregator(df = scores, agg_func=agg_func, k=cfg.k, imputer=imputer, df_unscored=df, scorer=scorer, **cfg.aggregator_kwargs)
    
    top_k_df = df.loc[result_dict["ranking"]]
    print("\nTop-k Ranked Items:")
    print(top_k_df)

    print(f"\nMethod: {result_dict['method']}")
    print(f"Sorted accesses: {result_dict['sorted_accesses']}")
    print(f"Random accesses: {result_dict['random_accesses']}")
    print(f"Total accesses: {result_dict['sorted_accesses'] + result_dict['random_accesses']}")


if __name__ == "__main__":
    main()
