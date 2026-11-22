"""Small numpy models for the audit (no scikit-learn dependency, so the unit tests run anywhere).

* `prepare_features`: per-view signal table -> design matrix. Non-negative heavy-tailed signals
  (gaps, residuals, spreads) go through log1p; confidence signals are used as they are. Missing
  values (e.g. no official ray poses, or no E_c re-pass) are imputed with the training median.
* `Ridge`: L2-regularised least squares on standardised features (closed form).
* `leave_one_group_out`: fit on all groups but one, predict the held-out group. Groups are
  datasets when at least two are present (the audit must transfer to unseen data), otherwise
  scenes (reported as within-dataset cross-validation).
"""

from __future__ import annotations

import numpy as np

RAW_FEATURES_PREFIX = "neg_logconf"


def prepare_features(table: dict, features: list[str]) -> np.ndarray:
    cols = []
    for f in features:
        x = np.asarray(table[f], dtype=np.float64)
        if not f.startswith(RAW_FEATURES_PREFIX):
            x = np.log1p(np.clip(x, 0.0, None))
        cols.append(x)
    return np.stack(cols, axis=1) if cols else np.zeros((len(next(iter(table.values()))), 0))


class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        keep = np.isfinite(y)
        X, y = X[keep], y[keep]
        self.median_ = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        Xi = self._impute(X)
        self.mean_ = Xi.mean(0) if len(Xi) else np.zeros(X.shape[1])
        self.std_ = Xi.std(0) if len(Xi) else np.ones(X.shape[1])
        self.std_ = np.where(self.std_ > 1e-12, self.std_, 1.0)
        Z = (Xi - self.mean_) / self.std_
        self.intercept_ = float(y.mean()) if len(y) else 0.0
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - self.intercept_)) if Z.shape[1] else np.zeros(0)
        return self

    def _impute(self, X):
        X = np.array(X, np.float64, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.broadcast_to(self.median_, X.shape)[bad]
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (self._impute(X) - self.mean_) / self.std_
        return Z @ self.coef_ + self.intercept_

    def to_dict(self, features: list[str]) -> dict:
        return {"alpha": self.alpha, "intercept": self.intercept_,
                "standardised_coefficients": dict(zip(features, map(float, self.coef_)))}


def leave_one_group_out(X: np.ndarray, y: np.ndarray, datasets: np.ndarray, scenes: np.ndarray,
                        make_model=lambda: Ridge(1.0)) -> tuple[np.ndarray, str, list]:
    """Out-of-group predictions for every row. Returns (pred, scheme, fitted models per fold)."""
    datasets = np.asarray(datasets)
    groups = datasets if len(np.unique(datasets)) >= 2 else np.asarray(scenes)
    scheme = "leave-one-dataset-out" if len(np.unique(datasets)) >= 2 else "leave-one-scene-out (one dataset only)"
    pred = np.full(len(y), np.nan)
    fitted = []
    for g in np.unique(groups):
        test = groups == g
        train = ~test & np.isfinite(y)
        if train.sum() < 5:
            continue
        m = make_model().fit(X[train], y[train])
        pred[test] = m.predict(X[test])
        fitted.append((str(g), m))
    return pred, scheme, fitted
