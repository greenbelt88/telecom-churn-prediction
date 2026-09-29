"""Preprocessing for the tree models and the logistic-regression pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler

from .config import SEED


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) column names."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    return num_cols, cat_cols


class TreePreprocessor:
    """Median-impute numerics and label-encode categoricals, fitted on training rows only.

    Missing categoricals become their own level ("nan"); labels unseen at fit time map to "__NA__".
    """

    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols, self.cat_cols = num_cols, cat_cols

    def fit(self, X: pd.DataFrame) -> "TreePreprocessor":
        self.imputer_ = SimpleImputer(strategy="median").fit(X[self.num_cols])
        self.encoders_ = {}
        for col in self.cat_cols:
            vocab = pd.concat([X[col].astype(str), pd.Series(["__NA__"])]).unique()
            self.encoders_[col] = LabelEncoder().fit(vocab)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(self.imputer_.transform(X[self.num_cols]), columns=self.num_cols, index=X.index)
        for col, le in self.encoders_.items():
            vals = X[col].astype(str)
            vals = vals.where(vals.isin(set(le.classes_)), "__NA__")
            out[col] = le.transform(vals)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def signed_log1p(A):
    return np.sign(A) * np.log1p(np.abs(A))


def as_str(A):
    return pd.DataFrame(A).fillna("missing").astype(str)


def make_logreg(num_cols: list[str], cat_cols: list[str], C: float = 0.1) -> Pipeline:
    """Logistic regression with preprocessing suited to a linear model.

    Numerics: median impute -> signed log -> standardise. Categoricals: one-hot (levels < 100 grouped).
    """
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(signed_log1p)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("str", FunctionTransformer(as_str)),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])
