"""Project-wide constants: paths, seed, column groups, tuned hyperparameters and business assumptions."""

from dataclasses import dataclass
from pathlib import Path

# Reproducibility & paths
SEED = 42
DATA_DIR = Path("telecom")
FIG_DIR = Path("figures")

# Plot colours (match the slide deck)
STAY_C = "#4C9BE8"
CHURN_C = "#E8694C"
GREEN_C = "#2E7D32"

# Columns
TARGET = "churn"
ID_COL = "Customer_ID"

DROP_COLS = [
    "Customer_ID",  # identifier with no predictive value; risks memorisation
    "recv_sms_Mean",  # undefined in the data dictionary
    "iwylis_vce_Mean",  # undefined in the data dictionary
    "ethnic",  # protected attribute, excluded on fairness grounds
]

# Model
# Result of the Optuna search in Section 6 (SEED=42, 30 trials, 5-fold CV at learning_rate=0.05)
TUNED_PARAMS = {
    "num_leaves": 39,
    "min_child_samples": 52,
    "colsample_bytree": 0.4971,
    "subsample": 0.9794,
    "reg_lambda": 16.0837,
}
FINAL_LR = 0.02
SEEDS = [SEED + i for i in range(5)]


# Business assumptions
@dataclass(frozen=True)
class Assumptions:
    """Campaign economics. Every value is a placeholder for Telecom's own figures."""

    monthly_revenue: float  # median monthly revenue per customer (from the data)
    lifetime_months: int = 24  # planning horizon for revenue saved
    success_rate: float = 0.30  # share of contacted would-be churners who stay (pilot measures this)
    offer_cost_months: float = 1.0  # offer cost, in months of the customer's revenue
    true_churn_rate: float = 0.03  # real monthly churn rate (the sample is ~50/50)
    customer_base: int = 100_000  # customers the results are scaled to

    @property
    def breakeven_probability(self) -> float:
        """Churn probability above which contacting a customer has positive expected value."""
        return self.offer_cost_months / (self.success_rate * self.lifetime_months)
