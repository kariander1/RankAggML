import pandas as pd

# Load your data
df1 = pd.read_csv(fr"outputs\imputer_for_agg\results.csv")  # <-- replace with your actual path
df2 = pd.read_csv(fr"outputs\imputer_for_agg\metrics.csv")  # <-- replace with your actual path

df = pd.merge(df1, df2, on=["Algorithm", "k"], how="inner")

algo_name_mapping = {
    "vgsales_combined_fagin": "Fagin",
    "vgsales_combined_threshold": "Threshold",
    "vgsales_combined_naive": "Naive",
    "vgsales_combined_nra": "NRA",
    "vgsales_combined_nra_impute": "NRA w/Imp",
    "nra_ref": "NRA",
    "basic": "NRA w/Basic Imp.",
    "heuristic": "NRA w/Heuristic Imp.",
    "ml": "NRA w/ML Imp.",
}
df["Algorithm"] = df["Algorithm"].replace(algo_name_mapping)
# algorithm_order = ["Naive", "Fagin", "Threshold", "NRA", "NRA w/Imp"]
algorithm_order = ["NRA", "NRA w/Basic Imp.", "NRA w/Heuristic Imp.", "NRA w/ML Imp."]
# Create a categorical column for sorting
df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=algorithm_order, ordered=True)

# Sort by algorithm and k
df = df.sort_values(by=["Algorithm", "k"])

# Clean algorithm names if needed

# Define the k values for each block
block1_ks = [1, 2, 3, 4, 5, 6]
block2_ks = [7, 8, 9, 10, 15, 20]
algorithms = df["Algorithm"].unique()

def format_entry(row):
    acc_emr = f"{row['set_accuracy']:.2f} / {row['exact_match_rate']:.2f}"
    sa_ra = f"{int(row['sorted_accesses'])} / {int(row['random_accesses'])}"
    return acc_emr, sa_ra

def latex_block(ks, block_title):

    col_headers = " & ".join([f"\\multicolumn{{2}}{{c}}{{$k = {k}$}}" for k in ks])
    sub_headers = " & ".join(["Acc / EMR & SA / RA"] * len(ks))
    cmrules = " ".join([f"\\cmidrule(r){{{2*i+2}-{2*i+3}}}" for i in range(len(ks))])
    
    table = "\\begin{tabular}{l|" + "cc" * len(ks) + "}\n\\toprule\n"
    table += "\\textbf{Algorithm} & " + col_headers + " \\\\\n"
    table += cmrules + "\n"
    table += "& " + sub_headers + " \\\\\n\\midrule\n"
    
    for algo in algorithm_order:
        row = f"{algo}"
        for k in ks:
            match = df[(df["Algorithm"] == algo) & (df["k"] == k)]
            if not match.empty:
                acc_emr, sa_ra = format_entry(match.iloc[0])
            else:
                acc_emr, sa_ra = "-- / --", "-- / --"
            row += f" & {acc_emr} & {sa_ra}"
        table += row + " \\\\\n"
    
    table += "\\bottomrule\n\\end{tabular}\n"
    return table

# Generate LaTeX code
block1 = latex_block(block1_ks, "$k = 1$ to $k = 6$")
block2 = latex_block(block2_ks, "$k = 7$ to $k = 10$, $15$, and $20$")

# Final wrap
latex_table = (
    "\\begin{table}[h]\n"
    "\\centering\n"
    "\\tiny\n"
    "\\caption{Comparison of algorithms at $p = 0$. For each $k$, we report Set Accuracy (Acc), Exact Match Rate (EMR), Sorted Accesses (SA), and Random Accesses (RA).}\n"
    "\\label{tab:sales_only_p0}\n\n"
    + block1 +
    "\n\\vspace{1em}\n\n"
    + block2 +
    "\\end{table}"
)

# Output or save
with open("latex_output_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved to latex_output_table.tex")
