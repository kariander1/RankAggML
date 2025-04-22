from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Config:
    output_path: str = field(default_factory=lambda: "outputs/imputer_for_agg/ml")
    dataset_name: str = "vgsales"
    dataset_kwargs: dict = field(default_factory=lambda: {
        'combine_sales': True,
        }
    )
    
    p_erase: float = 0
    imputer_name: str = "ml"
    imputer_kwargs: dict = field(default_factory=lambda: {
        'hidden_layer_sizes': (64, 64, 64, 64),
        'max_iter': 1000,
        'verbose': False,
        'use_cache': True,
    })
    scorer_kwargs: dict = field(default_factory=lambda: {
        'normalize': True,
        'columns_to_drop': ['earliest_year', 'num_platforms'],
        }
    )
    aggregator_name: str = "nra_w_impute"
    aggregator_kwargs: dict = field(default_factory=lambda: {})
    agg_function_name: str = "sum"
    
    k: int = field(default_factory=lambda: [1,2,3,4,5,6,7,8,9,10,15,20])
    seed: int = 42
