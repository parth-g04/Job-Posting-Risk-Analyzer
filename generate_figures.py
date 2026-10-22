"""
Generates high-resolution publication figures for academic reports and presentations.
Saves figures to reports/figures/.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR, METRICS_FILE


def generate_all_figures():
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not METRICS_FILE.exists():
        print("Metrics file not found. Run train_evaluate.py first.")
        return

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    models_data = data["models"]

    # 1. Model Comparison Bar Chart
    models = list(models_data.keys())
    precisions = [models_data[m]["test_metrics"]["fraud_precision"] for m in models]
    recalls = [models_data[m]["test_metrics"]["fraud_recall"] for m in models]
    f1s = [models_data[m]["test_metrics"]["fraud_f1"] for m in models]
    pr_aucs = [models_data[m]["test_metrics"]["pr_auc"] or 0 for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, precisions, width, label="Precision", color="#3498db")
    ax.bar(x - 0.5*width, recalls, width, label="Recall (Fraud)", color="#e74c3c")
    ax.bar(x + 0.5*width, f1s, width, label="F1-Score", color="#2ecc71")
    ax.bar(x + 1.5*width, pr_aucs, width, label="PR-AUC", color="#9b59b6")

    ax.set_title("Classifier Benchmark Comparison (Held-Out Test Set: 2,682 Listings)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(figures_dir / "model_comparison_bar.png", dpi=300)
    plt.close(fig)
    print("Saved: reports/figures/model_comparison_bar.png")

    # 2. Confusion Matrices Subplot (2x2 for the 4 ML models)
    target_models = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, name in enumerate(target_models):
        ax = axes[idx]
        cm = models_data[name]["test_metrics"]["confusion_matrix"]["raw"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"]
        )
        ax.set_title(f"{name}\n(F1: {models_data[name]['test_metrics']['fraud_f1']:.3f} | PR-AUC: {models_data[name]['test_metrics']['pr_auc']:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrices on Held-Out Test Set (130 Fraud, 2,552 Legit)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=300)
    plt.close(fig)
    print("Saved: reports/figures/confusion_matrices.png")

    # 3. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in target_models:
        pr = models_data[name].get("pr_curve")
        if pr:
            ax.plot(pr["recall"], pr["precision"], label=f"{name} (AUC = {pr['pr_auc']:.3f})", linewidth=2)

    ax.set_title("Precision-Recall Curves (Test Set Imbalance: 4.84% Fraud)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Detection Coverage)", fontsize=11)
    ax.set_ylabel("Precision (Reliability of Flag)", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=300)
    plt.close(fig)
    print("Saved: reports/figures/precision_recall_curves.png")

    # 4. ROC Curves
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.7)
    for name in target_models:
        roc = models_data[name].get("roc_curve")
        if roc:
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC = {roc['roc_auc']:.3f})", linewidth=2)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close(fig)
    print("Saved: reports/figures/roc_curves.png")

    # 5. Metadata Feature Ablation Comparison
    if "ablation_study" in data:
        ab = data["ablation_study"]
        to = ab["text_only"]
        cb = ab["combined"]

        metrics_names = ["Precision", "Recall", "F1-Score", "PR-AUC"]
        text_only_vals = [to["fraud_precision"], to["fraud_recall"], to["fraud_f1"], to["pr_auc"]]
        combined_vals = [cb["fraud_precision"], cb["fraud_recall"], cb["fraud_f1"], cb["pr_auc"]]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width/2, text_only_vals, width, label="Text-Only (TF-IDF alone)", color="#95a5a6")
        ax.bar(x + width/2, combined_vals, width, label="Combined (Text + Metadata)", color="#27ae60")

        ax.set_title("Feature Ablation Study: Impact of Adding Structured Metadata", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(figures_dir / "feature_ablation.png", dpi=300)
        plt.close(fig)
        print("Saved: reports/figures/feature_ablation.png")


if __name__ == "__main__":
    generate_all_figures()
