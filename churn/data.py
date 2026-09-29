"""Loading, feature engineering and splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, DROP_COLS, SEED, TARGET

ENGINEERED = [
    "overage_ratio",
    "mou_change_pct",
    "rev_change_pct",
    "recent_vs_life_mou",
    "recent_vs_life_rev",
    "eqp_to_tenure",
    "drop_rate",
]


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read Client.csv and Record.csv and inner-join them on Customer_ID."""
    client = pd.read_csv(Path(data_dir) / "Client.csv")
    record = pd.read_csv(Path(data_dir) / "Record.csv")
    print(f"Client : {client.shape}   Record : {record.shape}")
    return record.merge(client, on="Customer_ID", how="inner")


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features that trees cannot easily construct from single-column splits."""
    X = X.copy()
    X["overage_ratio"] = X["ovrmou_Mean"] / (X["mou_Mean"] + 1)
    X["mou_change_pct"] = X["change_mou"] / (X["avg3mou"] + 1)
    X["rev_change_pct"] = X["change_rev"] / (X["avg3rev"] + 1)
    X["recent_vs_life_mou"] = X["avg3mou"] / (X["avgmou"] + 1)
    X["recent_vs_life_rev"] = X["avg3rev"] / (X["avgrev"] + 1)
    X["eqp_to_tenure"] = X["eqpdays"] / (X["months"] * 30.4)
    X["drop_rate"] = X["drop_blk_Mean"] / (X["attempt_Mean"] + 1)
    return X


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifier / undefined / sensitive columns, engineer features, separate the target."""
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET])
    return add_features(X), df[TARGET].astype(int)


def split_60_20_20(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Stratified 60/20/20 train / validation / test split."""
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
