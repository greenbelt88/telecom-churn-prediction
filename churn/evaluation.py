"""Metrics, model comparison, bootstrap intervals, gains table, SHAP and segment churn rates."""

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import SEED


def top_decile_precision(y_true, proba, share: float = 0.10) -> float:
    """Share of real churners among the `share` highest-scored customers."""
    y_true = np.asarray(y_true)
    k = int(round(len(proba) * share))
    return y_true[np.argsort(-proba)[:k]].mean()


def evaluate(name, model, X_tr, y_tr, X_va, y_va, threshold: float = 0.5) -> dict:
    """Fit on (X_tr, y_tr), score on (X_va, y_va). Returns metrics plus the fitted model and scores."""
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "Model": name,
        "AUC-ROC": round(roc_auc_score(y_va, proba), 4),
        "Top-10% prec": round(top_decile_precision(y_va, proba), 4),
        "F1": round(f1_score(y_va, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_va, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_va, pred), 4),
        "_model": model,
        "_proba": proba,
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results]).sort_values(
        "AUC-ROC", ascending=False
    )


def test_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "top10": top_decile_precision(y_true, proba),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }


def bootstrap_ci(y_true, proba, n: int = 1000, seed: int = SEED) -> dict:
    """95% bootstrap intervals for AUC and top-decile precision."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    boot = np.array(
        [
            (roc_auc_score(y[i], proba[i]), top_decile_precision(y[i], proba[i]))
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
        ]
    )
    return {
        "auc": np.percentile(boot[:, 0], [2.5, 97.5]),
        "top10": np.percentile(boot[:, 1], [2.5, 97.5]),
    }


def gains_table(y_true, proba) -> pd.DataFrame:
    """Actual churn rate, lift and cumulative recall by predicted-risk decile (1 = riskiest)."""
    deciles = pd.qcut(pd.Series(proba).rank(method="first", ascending=False), 10, labels=range(1, 11))
    y = np.asarray(y_true)
    g = (
        pd.DataFrame({"decile": deciles.values, "churn": y})
        .groupby("decile", observed=True)["churn"]
        .agg(["mean", "sum", "size"])
    )
    g["lift"] = g["mean"] / y.mean()
    g["cum_recall"] = g["sum"].cumsum() / y.sum()
    return g.rename(columns={"mean": "precision", "sum": "churners", "size": "customers"})


def ensemble_shap(ensemble, X) -> np.ndarray:
    """SHAP values (log-odds) of a SeedEnsemble: the mean of its members' SHAP values."""

    def one(m):
        v = shap.TreeExplainer(m).shap_values(X)
        return v[1] if isinstance(v, list) else v  # older shap versions return [class0, class1]

    return np.mean([one(m) for m in ensemble.models_], axis=0)


def churn_rate_by(df: pd.DataFrame, col: str, bins=None, labels=None, right: bool = True) -> pd.DataFrame:
    """Churn rate and customer count per segment of `col` (binned when `bins` is given)."""
    key = df[col] if bins is None else pd.cut(df[col], bins=bins, labels=labels, right=right)
    return df.groupby(key, observed=True)["churn"].agg(rate="mean", customers="size")
