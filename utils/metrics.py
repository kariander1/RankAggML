import pandas as pd
from scipy.stats import kendalltau, spearmanr

def compute_accuracy_metrics(index1: pd.Index, index2: pd.Index) -> dict:
    """
    Compute set-based and order-based accuracy between two index vectors.
    
    Args:
        index1 (pd.Index): Ground truth or reference ranking.
        index2 (pd.Index): Predicted ranking.
        
    Returns:
        dict with:
            - set_accuracy: Intersection overlap (order-invariant)
            - exact_match_rate: Fraction of items at the same position
            - kendall_tau: Kendall’s tau rank correlation
            - spearman_rho: Spearman's rank correlation
    """
    # Ensure both indices are equal length and comparable
    if len(index1) != len(index2):
        raise ValueError("Indices must be of equal length to compare rankings")

    set1 = set(index1)
    set2 = set(index2)

    # Order-invariant set accuracy
    set_accuracy = len(set1 & set2) / len(set1)

    # Exact position match rate
    exact_match_rate = sum(i1 == i2 for i1, i2 in zip(index1, index2)) / len(index1)

    # Rank correlation measures
    rank1 = pd.Series(range(len(index1)), index=index1)
    rank2 = pd.Series(range(len(index2)), index=index2)

    # Align rankings to common elements
    common = list(set1 & set2)
    if len(common) < 2:
        tau, rho = float('nan'), float('nan')  # not defined on <2 elements
    else:
        tau = kendalltau(rank1[common], rank2[common]).correlation
        rho = spearmanr(rank1[common], rank2[common]).correlation

    return {
        "set_accuracy": set_accuracy,
        "exact_match_rate": exact_match_rate,
        "kendall_tau": tau,
        "spearman_rho": rho
    }


import os
import pandas as pd
from typing import Optional


def find_and_combine_csvs(base_dir: str, csv_filename: str, dir_pattern: Optional[str] = None, verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Search recursively through base_dir for subdirectories that match the given pattern,
    load CSV files with the specified name, and combine them into a single DataFrame.

    Args:
        base_dir (str): Root directory to search in.
        csv_filename (str): Target CSV filename to look for in each subdirectory.
        dir_pattern (Optional[str]): If provided, only include subdirectories whose names contain this pattern.
        verbose (bool): Whether to print loading messages.

    Returns:
        Optional[pd.DataFrame]: Combined DataFrame or None if no matches found.
    """
    combined_dfs = []

    for root, _, files in os.walk(base_dir):
        if csv_filename in files:
            relative_path = os.path.relpath(root, base_dir)
            algorithm_name = os.path.basename(relative_path)

            if dir_pattern and dir_pattern not in algorithm_name:
                continue

            full_path = os.path.join(root, csv_filename)
            try:
                df = pd.read_csv(full_path)
                df["Algorithm"] = algorithm_name
                combined_dfs.append(df)
                if verbose:
                    print(f"Loaded: {full_path} (Algorithm: {algorithm_name})")
            except Exception as e:
                if verbose:
                    print(f"Failed to load {full_path}: {e}")

    if not combined_dfs:
        print("No matching CSV files found.")
        return None

    return pd.concat(combined_dfs, ignore_index=True)


def main():
    base_dir = fr"outputs\imputer_for_agg"
    csv_filename = "results.csv"
    

    # Optional: Filter only directories that contain "nra"
    dir_pattern = None  # Set to something like "nra" to filter
    output_path = os.path.join(base_dir, f'{dir_pattern}_{csv_filename}')
    combined_df = find_and_combine_csvs(base_dir, csv_filename, dir_pattern=dir_pattern)

    if combined_df is not None:
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved to {output_path}")


if __name__ == "__main__":
    main()