from aggregation import naive, fagin, threshold, nra, nra_w_impute

def get_aggregator(name: str):
    name = name.lower()
    if name == "naive":
        return naive.aggregate
    elif name == "fagin":
        return fagin.aggregate
    elif name == "threshold":
        return threshold.aggregate
    elif name == "nra":
        return nra.aggregate
    elif name == "nra_w_impute":
        return nra_w_impute.aggregate
    else:
        raise ValueError(f"Unknown aggregation method: {name}")
