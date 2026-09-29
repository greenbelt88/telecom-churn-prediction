"""Plot helpers. Every figure used in the slides/README is saved to FIG_DIR."""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import CHURN_C, FIG_DIR, STAY_C


def set_style():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120
    FIG_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def rate_bars(
    ax,
    rates,
    overall=None,
    colors=None,
    title="",
    xlabel="",
    ylabel="Churn rate (%)",
    ylim=70,
    bold=True,
    fontsize=None,
    overall_label=True,
):
    """Bar chart of churn rates (0-1) with value labels and an optional dashed overall-rate line."""
    values = rates * 100
    labels = [str(i) for i in rates.index]
    ax.bar(labels, values, color=colors or STAY_C, edgecolor="white")
    if overall is not None:
        ax.axhline(
            overall * 100,
            color="grey",
            ls="--",
            lw=1,
            label=f"Overall ({overall:.0%})" if overall_label else None,
        )
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold" if bold else None, fontsize=fontsize)
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def model_comparison(res_df):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [CHURN_C] + [STAY_C] * (len(res_df) - 1)
    bars = ax.barh(res_df["Model"], res_df["AUC-ROC"], color=colors, edgecolor="white")
    ax.set_xlim(0.45, 0.8)
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="Random ranking (0.5)")
    for bar, val in zip(bars, res_df["AUC-ROC"], strict=True):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("AUC-ROC (validation set)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    return fig
