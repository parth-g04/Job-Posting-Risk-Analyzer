"""
Evaluation module with metrics focused on imbalanced classification:
Precision, Recall, F1-Score, Confusion Matrix, PR-AUC, and ROC-AUC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)


def compute_comprehensive_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute standard, balanced, and imbalanced metrics for binary fraud class."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn_pct": round(float(tn / total * 100), 2),
            "fp_pct": round(float(fp / total * 100), 2),
            "fn_pct": round(float(fn / total * 100), 2),
            "tp_pct": round(float(tp / total * 100), 2),
            "raw": cm.tolist()
        }
    }

    if y_prob is not None:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    return metrics


def analyze_threshold_tradeoffs(y_true, y_prob, thresholds=None):
    """Analyze precision and recall across various decision thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    results = []
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        p = precision_score(y_true, preds, pos_label=1, zero_division=0)
        r = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f = f1_score(y_true, preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        results.append({
            "threshold": round(thresh, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0])
        })
    return pd.DataFrame(results)


def get_pr_curve_data(y_true, y_prob):
    """Calculate precision-recall curve points."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }


def get_roc_curve_data(y_true, y_prob):
    """Calculate ROC curve points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }
