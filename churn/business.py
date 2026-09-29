"""Campaign economics: sample-scale campaign, prior-shift correction, targeting policies, pilot sizing."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Assumptions


# 9.1 Campaign on the sample as provided
def sample_campaign(
    a: Assumptions, precision: float, share: float, success_rate: float | None = None
) -> dict:
    """Contact `share` of the base at the given precision; revenue at the median customer value."""
    s = a.success_rate if success_rate is None else success_rate
    contacted = a.customer_base * share
    churners = contacted * precision
    retained = churners * s
    saved = retained * a.monthly_revenue * a.lifetime_months
    cost = contacted * a.monthly_revenue * a.offer_cost_months  # every contact costs, incl. non-churners
    return {
        "contacted": contacted,
        "churners reached": churners,
        "retained": retained,
        "revenue saved": saved,
        "offer cost": cost,
        "net": saved - cost,
        "ROI": (saved - cost) / cost,
    }


# 9.2 Correcting for the real churn rate
def to_real_prior(p, true_rate: float, sample_rate: float):
    """Prior-shift correction: sample-scale probability -> probability on a base with `true_rate` churn."""
    a = p * true_rate / sample_rate
    b = (1 - p) * (1 - true_rate) / (1 - sample_rate)
    return a / (a + b)


def real_weights(y, true_rate: float, sample_rate: float) -> np.ndarray:
    """Importance weights that make a sample look like a base with `true_rate` churn."""
    return np.where(np.asarray(y) == 1, true_rate / sample_rate, (1 - true_rate) / (1 - sample_rate))


def weighted_top_share_precision(proba, y, w, share: float = 0.10) -> float:
    """Precision among the top `share` of the (weighted) base, ranked by `proba`."""
    y = np.asarray(y)
    order = np.argsort(-proba)
    top = order[np.cumsum(w[order]) / w.sum() <= share]
    return (w[top] * y[top]).sum() / w[top].sum()


# 9.3 Targeting policies
class CampaignSimulator:
    """Evaluates targeting policies on a labelled set reweighted to a real churn rate.

    Revenue saved uses the customers' *actual* outcomes, not the model's predictions.
    """

    def __init__(self, proba, y, revenue, a: Assumptions, sample_rate: float):
        self.proba, self.y, self.revenue = np.asarray(proba), np.asarray(y), np.asarray(revenue)
        self.a, self.sample_rate = a, sample_rate

    def _setup(self, true_rate, offer_cost_months):
        w = real_weights(self.y, true_rate, self.sample_rate)
        p_real = to_real_prior(self.proba, true_rate, self.sample_rate)
        ev = self.revenue * (p_real * self.a.success_rate * self.a.lifetime_months - offer_cost_months)
        return w, ev

    def _top_share(self, score, w, q):
        o = np.argsort(-score)
        sel = np.zeros(len(score), bool)
        sel[o[np.cumsum(w[o]) / w.sum() <= q]] = True
        return sel

    def policies(self, true_rate=None, offer_cost_months=None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (weights, expected value per customer, {policy name: selection mask})."""
        true_rate = self.a.true_churn_rate if true_rate is None else true_rate
        c = self.a.offer_cost_months if offer_cost_months is None else offer_cost_months
        w, ev = self._setup(true_rate, c)
        return (
            w,
            ev,
            {
                "A. Top 10% by churn probability": self._top_share(self.proba, w, 0.10),
                "B. Top 10% by expected value": self._top_share(ev, w, 0.10),
                "C. Everyone with EV > 0": ev > 0,
            },
        )

    def outcome(self, sel, w, offer_cost_months=None) -> dict:
        """Campaign result per `customer_base` customers."""
        c = self.a.offer_cost_months if offer_cost_months is None else offer_cost_months
        k = self.a.customer_base / w.sum()
        ws, ys, rs = w[sel], self.y[sel], self.revenue[sel]
        contacted = ws.sum() * k
        churners = (ws * ys).sum() * k
        saved = (ws * ys * rs).sum() * k * self.a.success_rate * self.a.lifetime_months
        cost = (ws * rs).sum() * k * c
        return {
            "contacted": contacted,
            "churners reached": churners,
            "precision": churners / max(contacted, 1e-9),
            "revenue saved": saved,
            "offer cost": cost,
            "net": saved - cost,
            "ROI": (saved - cost) / cost if cost else np.nan,
        }

    def compare(self, true_rate=None, offer_cost_months=None) -> pd.DataFrame:
        w, _, pols = self.policies(true_rate, offer_cost_months)
        return pd.DataFrame({n: self.outcome(sel, w, offer_cost_months) for n, sel in pols.items()}).T

    def profit_curve(self, true_rate) -> tuple[np.ndarray, np.ndarray]:
        """(share of base contacted in %, cumulative net per base) when contacting in expected-value order."""
        w, ev, _ = self.policies(true_rate)
        o = np.argsort(-ev)
        k = self.a.customer_base / w.sum()
        net = (
            w[o]
            * (
                self.y[o] * self.revenue[o] * self.a.success_rate * self.a.lifetime_months
                - self.revenue[o] * self.a.offer_cost_months
            )
            * k
        )
        return np.cumsum(w[o]) / w.sum() * 100, np.cumsum(net)


# 9.4 Pilot sizing
def pilot_size(p_control: float, relative_reduction: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers per arm for a two-proportion test detecting `relative_reduction` in churn."""
    p_treated = p_control * (1 - relative_reduction)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    var = p_control * (1 - p_control) + p_treated * (1 - p_treated)
    return int(np.ceil(z**2 * var / (p_control - p_treated) ** 2))


# Formatting
def money(x: float) -> str:
    """Format as $1,234 or -$1,234."""
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def fmt_campaign(t: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of a policy / campaign table."""
    out = t.copy()
    for c in out.columns:
        if c in ("contacted", "churners reached", "retained"):
            out[c] = out[c].map("{:,.0f}".format)
        elif c in ("revenue saved", "offer cost", "net"):
            out[c] = out[c].map(money)
        elif c == "precision":
            out[c] = out[c].map("{:.1%}".format)
        elif c == "ROI":
            out[c] = out[c].map("{:.1f}x".format)
    return out
