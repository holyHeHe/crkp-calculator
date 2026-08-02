from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.validation import check_array


def check_array_compatible(X):
    """Support both old and new scikit-learn versions."""
    try:
        return check_array(
            X,
            dtype=float,
            ensure_all_finite=True
        )
    except TypeError:
        return check_array(
            X,
            dtype=float,
            force_all_finite=True
        )

class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric feature to fold-specific lower/upper quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_checked = check_array_compatible(X)
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1.")
        self.lower_bounds_ = np.quantile(X_checked, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(X_checked, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        X_checked = check_array_compatible(X)
        return np.clip(X_checked, self.lower_bounds_, self.upper_bounds_)
