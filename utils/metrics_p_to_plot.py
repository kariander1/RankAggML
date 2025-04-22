import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

def plot_set_accuracy(csv_path, smooth=True):
    # Load the CSV
    df = pd.read_csv(csv_path)
    df["Algorithm"] = df["Algorithm"].astype(str).str.split('_').str[-1]
    p = float(csv_path.split("_")[-3].split(".")[0] + '.' + csv_path.split("_")[-2].split(".")[0])
    
    color_map = {
        "drop": "#1f77b4",
        "basic": "#ff7f0e",
        "heuristic": "#2ca02c",
        "ml": "#d62728"
    }
    all_algorithms = list(color_map.keys())

    # Ensure all combinations of Algorithm and k are present
    all_k = sorted(df["k"].unique())
    full_index = pd.MultiIndex.from_product([all_algorithms, all_k], names=["Algorithm", "k"])
    df = df.set_index(["Algorithm", "k"]).reindex(full_index).reset_index()
    df["set_accuracy"] = df["set_accuracy"].fillna(0)

    # Prepare pivot for overlay logic
    pivot_df = df.pivot(index="k", columns="Algorithm", values="set_accuracy")
    winner_by_k = pivot_df.idxmax(axis=1)  # best algorithm per k
    regions = []
    current_algo = None
    start_k = None

    for k in pivot_df.index:
        algo = winner_by_k.loc[k]
        if algo != current_algo:
            if current_algo is not None:
                regions.append((start_k, k, current_algo))
            current_algo = algo
            start_k = k
    # Add the final region
    if current_algo is not None:
        regions.append((start_k, pivot_df.index.max(), current_algo))

    # Begin plot
    plt.figure(figsize=(6, 6))

    # Add transparent overlays
    for start_k, end_k, algo in regions:
        color = color_map.get(algo, "#cccccc")
        plt.axvspan(start_k, end_k, color=color, alpha=0.2)

    # Plot curves
    for algorithm, group in df.groupby("Algorithm"):
        color = color_map.get(algorithm, None)
        group = group.sort_values("k")
        x = group["k"].values
        y = group["set_accuracy"].values

        if smooth and len(x) > 3:
            x_new = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=2)
            y_smooth = spline(x_new)
            plt.plot(x_new, y_smooth, label=algorithm, color=color, lw=2.5)
        else:
            plt.plot(x, y, marker='o', label=algorithm, color=color, lw=2.5)

    plt.xlabel("k", fontsize=20)
    plt.ylabel("Set Accuracy", fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(1, 50)
    plt.title(fr"$p={p}$", fontsize=26)
    plt.legend(title="Imputer", fontsize=16, title_fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"outputs/imputation/p_{int(p*10)}_set_accuracy.pdf")
    # plt.show()
    plt.close()
# Example usage:
for p in [1,2,3,4,5,6,7,8]:
    plot_set_accuracy(fr"outputs/imputation/p_0_{p}_metrics.csv")
