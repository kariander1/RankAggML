import pandas as pd
import numpy as np
import itertools
import os
import hashlib
import joblib
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from typing import Union
from imputation.basic_imputer import BasicImputer


class MLSalesImputer:
    def __init__(self, df: pd.DataFrame, model_class=MLPRegressor, use_cache=False, cache_dir=".cache/ml_sales_imputer", **kwargs):
        """
        Train models to predict each sales column from all subsets of the others.
        If use_cache is True, load/save models from disk to avoid retraining.
        """
        self.sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        self.models = {}         # {('EU_Sales', ('NA_Sales',)): model, ...}
        self.model_scores = {}   # {('EU_Sales', ('NA_Sales',)): R² score, ...}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.model_class = model_class
        self.model_kwargs = kwargs
        self.basic_imputer = BasicImputer(df, **kwargs)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        df = df.copy()
        complete_rows = df.dropna(subset=self.sales_cols)

        for target_col in self.sales_cols:
            features = [col for col in self.sales_cols if col != target_col]
            features = sorted(features)  # ensure consistent order

            for k in range(1, len(features) + 1):
                for subset in itertools.combinations(features, k):
                    X = complete_rows[list(subset)]
                    y = complete_rows[target_col]

                    cache_key = self._make_cache_key(target_col, subset, X, y)

                    if self.use_cache and self._has_cache(cache_key):
                        model, r2 = self._load_from_cache(cache_key)
                        print(f"[Cache] Loaded {target_col} ← {subset} | R² = {r2:.4f}")
                    else:
                        model = model_class(**kwargs)
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        r2 = r2_score(y, y_pred)
                        print(f"[Train] {target_col} ← {subset} | R² = {r2:.4f}")
                        if self.use_cache:
                            self._save_to_cache(cache_key, model, r2)

                    self.models[(target_col, subset)] = model
                    self.model_scores[(target_col, subset)] = r2

    def _make_cache_key(self, target_col, feature_subset, X, y):
        """Generate a unique cache key based on target, subset, model, and data."""
        key_string = f"{target_col}_{'_'.join(feature_subset)}_{self.model_class.__name__}"
        data_hash = hashlib.md5(pd.concat([X, y], axis=1).to_csv(index=False).encode()).hexdigest()
        return f"{key_string}_{data_hash}"

    def _has_cache(self, key):
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.joblib"))

    def _save_to_cache(self, key, model, r2):
        joblib.dump((model, r2), os.path.join(self.cache_dir, f"{key}.joblib"))

    def _load_from_cache(self, key):
        return joblib.load(os.path.join(self.cache_dir, f"{key}.joblib"))

    def impute(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Public entry point to impute either a DataFrame or a Series.
        """
        if isinstance(data, pd.Series):
            return self.impute_series(data)
        return self.impute_dataframe(data)

    def impute_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing sales values in a full DataFrame.
        Each missing value is imputed using the model with the highest R² score applicable to that row.
        """
        df = df.copy()
        
        for col in self.sales_cols:
            if col not in df.columns:
                df[col] = np.nan

        for target_col in self.sales_cols:
            missing_mask = df[target_col].isnull()
            if not missing_mask.any():
                continue

            # Store candidate imputations and their R² scores
            candidate_preds = pd.Series(index=df.index, dtype=float)
            best_r2s = pd.Series(index=df.index, dtype=float)

            for feature_subset in self._get_feature_subsets(target_col):
                feature_subset = tuple(sorted(feature_subset))  # ensure consistent order
                model = self.models.get((target_col, feature_subset))
                r2 = self.model_scores.get((target_col, feature_subset), -np.inf)

                if model is None:
                    continue

                subset_mask = df[list(feature_subset)].notnull().all(axis=1)
                applicable_mask = missing_mask & subset_mask

                if applicable_mask.any():
                    X_missing = df.loc[applicable_mask, list(feature_subset)]
                    predictions = model.predict(X_missing)

                    for idx, pred in zip(X_missing.index, predictions):
                        # Only use this prediction if it has higher r2 than previous
                        if pd.isna(candidate_preds.loc[idx]) or r2 > best_r2s.loc[idx]:
                            candidate_preds.loc[idx] = pred
                            best_r2s.loc[idx] = r2

            # Apply the best predictions
            df.loc[candidate_preds.notnull(), target_col] = candidate_preds[candidate_preds.notnull()]

        # Fallback for any remaining NaNs (e.g., non-sales columns or unhandled cases)
        df = self.basic_imputer.impute(df)

        return df

    def impute_series(self, series: pd.Series) -> pd.Series:
        series = series.copy()
        known_cols = tuple(sorted(set(col for col in self.sales_cols if pd.notnull(series[col]))))
        if len(known_cols) > 0:
            for target_col in self.sales_cols:
                if pd.notnull(series[target_col]):
                    continue

                model = self.models[(target_col, known_cols)]
                r2 = self.model_scores[(target_col, known_cols)]
                # if r2> 0.97:  # only use models with reasonable R² score
                X = series[list(known_cols)].to_frame().T
                prediction = model.predict(X)[0]
                # series[target_col] = prediction
                # series[target_col] = prediction*(1+(1-r2))  # scale by R² score
                series[target_col] = prediction * r2  # scale by R² score

        return series

    
    def _get_feature_subsets(self, target_col):
        """Return feature subsets sorted by length descending (prefer more features)."""
        subsets = [
            key[1] for key in self.models.keys()
            if key[0] == target_col
        ]
        return sorted(subsets, key=lambda x: -len(x))  # longest (most features) first

    def get_confidence(self, target_col, feature_subset):
        """Return the R² score of the model used to predict target_col from feature_subset."""
        return self.model_scores.get((target_col, feature_subset), None)
