"""
evaluation.py
-------------
Model evaluation utilities: metrics, confusion matrices, comparison plots.

All metrics come from actually running the trained models.
No results are fabricated.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripts/notebooks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, classes=None) -> dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true  : array-like of true integer/string labels
    y_pred  : array-like of predicted labels
    classes : list of class names (optional, for labelled output)

    Returns
    -------
    dict with accuracy, macro_f1, weighted_f1, macro_precision,
    macro_recall, weighted_precision, weighted_recall, classification_report
    """
    acc = accuracy_score(y_true, y_pred)
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if classes is not None:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=classes, zero_division=0
        )
    else:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, zero_division=0
        )
    return metrics


def evaluate_model(model, X, y_true, label_encoder, split_name="Test") -> dict:
    """
    Evaluate a trained model on a dataset split.

    Parameters
    ----------
    model        : trained estimator
    X            : feature matrix
    y_true       : integer or string true labels
    label_encoder: LabelEncoder to map integers → class names
    split_name   : label for display ('Validation', 'Test', etc.)

    Returns
    -------
    dict of evaluation metrics.
    """
    y_pred = model.predict(X)
    classes = label_encoder.classes_
    metrics = compute_metrics(y_true, y_pred, classes=classes)
    metrics["split"] = split_name
    return metrics, y_pred


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize=(16, 14),
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalise by row (true class counts)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Normalised Rate"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Model comparison chart ────────────────────────────────────────────────────

def plot_model_comparison(
    comparison_dict: dict,
    metric: str = "weighted_f1",
    title: str = "Model Comparison",
    save_path: str = None,
) -> plt.Figure:
    """
    Bar chart comparing multiple models on a single metric.

    Parameters
    ----------
    comparison_dict : {model_name: metrics_dict}
    metric          : key from compute_metrics() to compare
    """
    names = list(comparison_dict.keys())
    values = [comparison_dict[n].get(metric, 0) for n in names]

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors[: len(names)], edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multi_metric_comparison(
    comparison_dict: dict,
    metrics=("accuracy", "macro_f1", "weighted_f1"),
    save_path: str = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing multiple models across multiple metrics.
    """
    models = list(comparison_dict.keys())
    n_metrics = len(metrics)
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, metric in enumerate(metrics):
        vals = [comparison_dict[m].get(metric, 0) for m in models]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric.replace("_", " ").title(), color=palette[i])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Save text report ──────────────────────────────────────────────────────────

def save_classification_report(report: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(report)


def save_comparison_csv(comparison_dict: dict, path: str) -> None:
    """Save model comparison as CSV."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metric_keys = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + metric_keys)
        for name, metrics in comparison_dict.items():
            row = [name] + [f"{metrics.get(k, 0):.4f}" for k in metric_keys]
            writer.writerow(row)
