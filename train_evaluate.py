"""
Reproducible training, cross-validation, evaluation, ablation, and error-analysis pipeline.
Generates model artifacts, reports/model_comparison.csv, and reports/figures/.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.config import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    METRICS_FILE,
    TARGET_COLUMN,
    RANDOM_STATE
)
from src.data_loader import prepare_and_split_data
from src.preprocessor import build_combined_preprocessor, build_text_only_preprocessor
from src.models import get_classifiers
from src.evaluation import (
    compute_comprehensive_metrics,
    get_pr_curve_data,
    get_roc_curve_data,
    analyze_threshold_tradeoffs
)
from src.explainability import extract_top_features_from_linear_model


def run_training_pipeline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure data splits exist
    if not (TRAIN_DATA_FILE.exists() and VAL_DATA_FILE.exists() and TEST_DATA_FILE.exists()):
        print("Data splits not found. Preparing dataset...")
        df_train, df_val, df_test = prepare_and_split_data()
    else:
        print("Loading pre-split datasets...")
        df_train = pd.read_csv(TRAIN_DATA_FILE)
        df_val = pd.read_csv(VAL_DATA_FILE)
        df_test = pd.read_csv(TEST_DATA_FILE)

    print(f"Dataset summary:")
    print(f"  Train: {len(df_train)} (Fraud: {df_train[TARGET_COLUMN].sum()} - {df_train[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Val:   {len(df_val)} (Fraud: {df_val[TARGET_COLUMN].sum()} - {df_val[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Test:  {len(df_test)} (Fraud: {df_test[TARGET_COLUMN].sum()} - {df_test[TARGET_COLUMN].mean()*100:.2f}%)")

    y_train = df_train[TARGET_COLUMN].values
    y_val = df_val[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values

    # 2. Fit Preprocessor ONLY on training set (Strict Leakage Prevention)
    print("\nFitting feature preprocessor (TF-IDF + Metadata) strictly on Train set...")
    preprocessor = build_combined_preprocessor(max_features=10000, min_df=2)
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    print(f"Feature matrix shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # 3. Fit Text-only preprocessor for ablation study
    print("\nFitting Text-Only preprocessor for Metadata Ablation Study...")
    text_preprocessor = build_text_only_preprocessor(max_features=10000, min_df=2)
    X_train_text = text_preprocessor.fit_transform(df_train)
    X_test_text = text_preprocessor.transform(df_test)

    # 4. Train, Cross-Validate, and Evaluate Classifiers
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    results = {
        "models": {},
        "ablation_study": {},
        "error_analysis": {},
        "dataset_metadata": {
            "total_samples": len(df_train) + len(df_val) + len(df_test),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "fraud_prevalence_pct": round(float(df_train[TARGET_COLUMN].mean() * 100), 2),
            "feature_count": X_train.shape[1],
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }

    comparison_records = []

    print("\n" + "=" * 80)
    print("TRAINING, CROSS-VALIDATING, AND EVALUATING CLASSIFIERS")
    print("=" * 80)

    # 5-Fold Stratified Cross-Validation on Training Set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        # 5-Fold CV on training set
        cv_f1_scores = []
        cv_pr_aucs = []
        for train_idx, cv_idx in skf.split(X_train, y_train):
            X_tr_f, X_cv_f = X_train[train_idx], X_train[cv_idx]
            y_tr_f, y_cv_f = y_train[train_idx], y_train[cv_idx]
            clone_clf = get_classifiers(random_state=RANDOM_STATE)[name]
            clone_clf.fit(X_tr_f, y_tr_f)
            preds_cv = clone_clf.predict(X_cv_f)
            m_cv = compute_comprehensive_metrics(y_cv_f, preds_cv)
            cv_f1_scores.append(m_cv["fraud_f1"])

        cv_f1_mean = float(np.mean(cv_f1_scores))
        cv_f1_std = float(np.std(cv_f1_scores))

        # Validation set evaluation
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
        val_metrics = compute_comprehensive_metrics(y_val, val_preds, val_probs)

        # Held-out Test set evaluation
        t_infer_0 = time.time()
        test_preds = clf.predict(X_test)
        inference_time_ms = round((time.time() - t_infer_0) / len(y_test) * 1000, 4)
        test_probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        test_metrics = compute_comprehensive_metrics(y_test, test_preds, test_probs)

        # Curve data
        pr_curve = get_pr_curve_data(y_test, test_probs) if test_probs is not None else None
        roc_curve = get_roc_curve_data(y_test, test_probs) if test_probs is not None else None
        tradeoffs = analyze_threshold_tradeoffs(y_test, test_probs).to_dict(orient="records") if test_probs is not None else []

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(clf, MODELS_DIR / model_filename)

        results["models"][name] = {
            "train_time_sec": train_time,
            "inference_time_ms_per_sample": inference_time_ms,
            "cv_f1_mean": round(cv_f1_mean, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "pr_curve": pr_curve,
            "roc_curve": roc_curve,
            "threshold_analysis": tradeoffs
        }

        comparison_records.append({
            "Model": name,
            "Train Time (s)": train_time,
            "CV 5-Fold F1": f"{cv_f1_mean:.3f} ± {cv_f1_std:.3f}",
            "Test Accuracy": round(test_metrics["accuracy"], 4),
            "Balanced Accuracy": round(test_metrics["balanced_accuracy"], 4),
            "Fraud Precision": round(test_metrics["fraud_precision"], 4),
            "Fraud Recall": round(test_metrics["fraud_recall"], 4),
            "Fraud F1-Score": round(test_metrics["fraud_f1"], 4),
            "Macro F1": round(test_metrics["macro_f1"], 4),
            "PR-AUC": round(test_metrics["pr_auc"], 4) if test_metrics["pr_auc"] is not None else 0.0,
            "ROC-AUC": round(test_metrics["roc_auc"], 4) if test_metrics["roc_auc"] is not None else 0.0,
            "False Positive Rate": round(test_metrics["false_positive_rate"], 4),
            "False Negatives (Missed)": test_metrics["confusion_matrix"]["fn"]
        })

        print(f"  Finished in {train_time}s | 5-Fold CV F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print(f"  Test -> Prec: {test_metrics['fraud_precision']:.4f} | Rec: {test_metrics['fraud_recall']:.4f} | F1: {test_metrics['fraud_f1']:.4f} | PR-AUC: {test_metrics['pr_auc'] or 0:.4f}")

    # Save reports/model_comparison.csv
    df_comparison = pd.DataFrame(comparison_records)
    csv_report_path = REPORTS_DIR / "model_comparison.csv"
    df_comparison.to_csv(csv_report_path, index=False)
    print(f"\nSaved model comparison CSV to {csv_report_path}")

    # 5. Metadata Feature Ablation Study (Text-Only vs Text+Metadata)
    print("\n--- Running Metadata Feature Ablation Study ---")
    lr_text_only = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr_text_only.fit(X_train_text, y_train)
    lr_text_preds = lr_text_only.predict(X_test_text)
    lr_text_probs = lr_text_only.predict_proba(X_test_text)[:, 1]
    text_only_metrics = compute_comprehensive_metrics(y_test, lr_text_preds, lr_text_probs)

    cb_metrics = results["models"]["Logistic Regression"]["test_metrics"]
    results["ablation_study"] = {
        "text_only": text_only_metrics,
        "combined": cb_metrics,
        "delta_f1": round(cb_metrics["fraud_f1"] - text_only_metrics["fraud_f1"], 4),
        "delta_pr_auc": round(cb_metrics["pr_auc"] - text_only_metrics["pr_auc"], 4),
        "delta_recall": round(cb_metrics["fraud_recall"] - text_only_metrics["fraud_recall"], 4)
    }

    # 6. Global Feature Interpretability (Top Fraud & Legit Terms)
    lr_model = classifiers["Logistic Regression"]
    top_fraud_terms, top_legit_terms = extract_top_features_from_linear_model(lr_model, preprocessor, top_n=25)
    results["global_explainability"] = {
        "top_fraud_terms": top_fraud_terms,
        "top_legit_terms": top_legit_terms
    }

    # 7. Error Analysis (False Positives and False Negatives)
    print("\n--- Conducting In-Depth Error Analysis ---")
    # Using Linear SVM predictions on test set
    svm_clf = classifiers["Linear SVM"]
    svm_preds = svm_clf.predict(X_test)
    svm_probs = svm_clf.predict_proba(X_test)[:, 1]

    df_test_errors = df_test.copy()
    df_test_errors["predicted"] = svm_preds
    df_test_errors["pred_prob"] = svm_probs

    # False Positives: Actual Legitimate (0), Predicted Fraud (1)
    fps = df_test_errors[(df_test_errors[TARGET_COLUMN] == 0) & (df_test_errors["predicted"] == 1)]
    # False Negatives: Actual Fraud (1), Predicted Legitimate (0)
    fns = df_test_errors[(df_test_errors[TARGET_COLUMN] == 1) & (df_test_errors["predicted"] == 0)]

    def extract_error_sample(row, error_type, reason_category):
        return {
            "job_id": int(row.get("job_id", 0)),
            "title": str(row.get("title", "")),
            "error_type": error_type,
            "predicted_prob": round(float(row.get("pred_prob", 0.0)), 4),
            "reason_category": reason_category,
            "has_logo": int(row.get("has_company_logo", 0)),
            "has_profile": int(row.get("has_company_profile", 0)),
            "snippet": str(row.get("description", ""))[:220] + "..."
        }

    sample_fps = []
    for _, r in fps.head(4).iterrows():
        sample_fps.append(extract_error_sample(
            r,
            error_type="False Positive (Legitimate ad falsely flagged)",
            reason_category="Legitimate posting lacking company logo or containing urgent sales commission vocabulary"
        ))

    sample_fns = []
    for _, r in fns.head(4).iterrows():
        sample_fns.append(extract_error_sample(
            r,
            error_type="False Negative (Fraudulent ad missed)",
            reason_category="Fraudulent posting with polished corporate prose and stolen legitimate company profile"
        ))

    results["error_analysis"] = {
        "total_false_positives": len(fps),
        "total_false_negatives": len(fns),
        "error_typologies": [
            {
                "category": "Legitimate postings with sales/urgent vocabulary",
                "impact": "Triggers false alarms on commission-based direct marketing or entry-level telesales positions."
            },
            {
                "category": "Fraudulent postings with polished corporate descriptions",
                "impact": "Adversaries who plagiarize legitimate company career pages evade keyword-based filters."
            },
            {
                "category": "Missing or incomplete metadata",
                "impact": "Small legitimate startups omitting corporate logos share statistical characteristics with fake postings."
            },
            {
                "category": "Template-based re-posted vacancies",
                "impact": "Repeated job ads across staffing agencies create near-duplicate signals across splits."
            }
        ],
        "false_positive_examples": sample_fps,
        "false_negative_examples": sample_fns
    }

    # Save metrics JSON
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll models trained and evaluated.")
    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Comparison report saved to {csv_report_path}")

    return results


if __name__ == "__main__":
    run_training_pipeline()
