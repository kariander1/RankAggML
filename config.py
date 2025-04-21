from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Config:
    output_path: str = field(default_factory=lambda: "outputs")
    dataset_name: str = "vgsales"
    imputer_name: str = "ml"
    dataset_kwargs: dict = field(default_factory=lambda: {
        'combine_sales': True
    })
    imputer_kwargs: dict = field(default_factory=lambda: {
        'hidden_layer_sizes': (64, 64, 64, 64),
        'max_iter': 1000,
        'verbose': False,
        'use_cache': True,
    })
    scorer_kwargs: dict = field(default_factory=lambda: {
        
    })
    aggregator_name: str = "nra_w_impute"
    aggregator_kwargs: dict = field(default_factory=lambda: {
        'complete_scores': True
        })
    agg_function_name: str = "sum"
    k: int = 7
    p_erase: float = 0
    seed: int = 42
