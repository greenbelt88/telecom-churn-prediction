"""LightGBM helpers: cross-validated early stopping, Optuna search, and the seed-averaged final model."""

import lightgbm as lgb
import numpy as np
import optuna

from .config import FINAL_LR, SEED

# Settings shared by every LightGBM fit. subsample_freq=1 is required for `subsample` to take effect.
LGB_FIXED = dict(
    objective="binary",
    metric="auc",
    subsample_freq=1,
    seed=SEED,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)


def make_dataset(X, y) -> lgb.Dataset:
    return lgb.Dataset(X, y, free_raw_data=False, params={"feature_pre_filter": False})


def cv_auc(params: dict, data: lgb.Dataset, folds: list, lr: float) -> tuple[float, int]:
    """Cross-validated AUC with early stopping. Returns (best mean AUC, number of rounds)."""
    res = lgb.cv(
        {**LGB_FIXED, **params, "learning_rate": lr},
        data,
        num_boost_round=5000,
        folds=folds,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return res["valid auc-mean"][-1], len(res["valid auc-mean"])


def optuna_search(data: lgb.Dataset, folds: list, n_trials: int = 30, lr: float = 0.05):
    """TPE search over complexity and regularisation parameters; each trial is scored by `cv_auc`."""

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
        auc, rounds = cv_auc(params, data, folds, lr=lr)
        trial.set_user_attr("rounds", rounds)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


def tuned_lgbm(params: dict, n_rounds: int, seed: int = SEED, lr: float = FINAL_LR) -> lgb.LGBMClassifier:
    """A LightGBM classifier with the tuned parameters and a fixed number of trees."""
    return lgb.LGBMClassifier(
        **params,
        n_estimators=n_rounds,
        learning_rate=lr,
        subsample_freq=1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )


class SeedEnsemble:
    """Average of tuned LightGBM models that differ only in random seed.

    Averaging is done on the log-odds scale, so per-model SHAP values (also log-odds) average exactly.
    """

    def __init__(self, params: dict, n_rounds: int, seeds: list[int]):
        self.params, self.n_rounds, self.seeds = params, n_rounds, seeds

    def fit(self, X, y) -> "SeedEnsemble":
        self.models_ = [tuned_lgbm(self.params, self.n_rounds, seed=s).fit(X, y) for s in self.seeds]
        return self

    def predict_margin(self, X) -> np.ndarray:
        return np.mean([m.predict(X, raw_score=True) for m in self.models_], axis=0)

    def predict_proba(self, X) -> np.ndarray:
        p = 1 / (1 + np.exp(-self.predict_margin(X)))
        return np.column_stack([1 - p, p])
