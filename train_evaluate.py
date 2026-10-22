"""
Training and evaluation pipeline for the 4 classifiers.
Adheres strictly to leakage prevention protocols and saves all model artifacts.
"""
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from src.config import (
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    MODELS_DIR,
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
    print(f"  Train: {len(df_train)} (Fraud: {df_train[TARGET_COLUMN].sum()})")
    print(f"  Val:   {len(df_val)} (Fraud: {df_val[TARGET_COLUMN].sum()})")
    print(f"  Test:  {len(df_test)} (Fraud: {df_test[TARGET_COLUMN].sum()})")

    y_train = df_train[TARGET_COLUMN].values
    y_val = df_val[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values

    # 2. Fit Preprocessor ONLY on training set (Leakage Prevention)
    print("\nFitting feature preprocessor (TF-IDF + Metadata) on Train set...")
    preprocessor = build_combined_preprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    print(f"Feature matrix shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # 3. Fit Text-only preprocessor for ablation study
    print("\nFitting Text-Only preprocessor for Metadata Ablation Study...")
    text_preprocessor = build_text_only_preprocessor()
    X_train_text = text_preprocessor.fit_transform(df_train)
    X_val_text = text_preprocessor.transform(df_val)
    X_test_text = text_preprocessor.transform(df_test)

    # 4. Train and Compare the 4 Classifiers
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    results = {
        "models": {},
        "ablation_study": {},
        "dataset_metadata": {
            "total_samples": len(df_train) + len(df_val) + len(df_test),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "fraud_prevalence_pct": round(float(df_train[TARGET_COLUMN].mean() * 100), 2)
        }
    }

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATING THE 4 CLASSIFIERS")
    print("=" * 80)

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)
        print(f"  Finished in {train_time}s")

        # Validation set evaluation
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
        val_metrics = compute_comprehensive_metrics(y_val, val_preds, val_probs)

        # Test set evaluation
        test_preds = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        test_metrics = compute_comprehensive_metrics(y_test, test_preds, test_probs)

        # Curve data for interactive visualizer
        pr_curve = get_pr_curve_data(y_test, test_probs) if test_probs is not None else None
        roc_curve = get_roc_curve_data(y_test, test_probs) if test_probs is not None else None

        # Threshold tradeoffs
        tradeoffs = analyze_threshold_tradeoffs(y_test, test_probs).to_dict(orient="records") if test_probs is not None else []

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(clf, MODELS_DIR / model_filename)

        results["models"][name] = {
            "train_time_sec": train_time,
            "artifact_path": str(MODELS_DIR / model_filename),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "pr_curve": pr_curve,
            "roc_curve": roc_curve,
            "threshold_analysis": tradeoffs
        }

        print(f"  Val  -> Precision: {val_metrics['fraud_precision']:.4f} | Recall: {val_metrics['fraud_recall']:.4f} | F1: {val_metrics['fraud_f1']:.4f} | PR-AUC: {val_metrics['pr_auc'] or 0:.4f}")
        print(f"  Test -> Precision: {test_metrics['fraud_precision']:.4f} | Recall: {test_metrics['fraud_recall']:.4f} | F1: {test_metrics['fraud_f1']:.4f} | PR-AUC: {test_metrics['pr_auc'] or 0:.4f}")

    # 5. Metadata Ablation Study (Text-Only vs Text+Metadata with Logistic Regression)
    print("\n--- Running Metadata Feature Ablation ---")
    from sklearn.linear_model import LogisticRegression
    lr_text_only = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr_text_only.fit(X_train_text, y_train)
    lr_text_preds = lr_text_only.predict(X_test_text)
    lr_text_probs = lr_text_only.predict_proba(X_test_text)[:, 1]
    text_only_metrics = compute_comprehensive_metrics(y_test, lr_text_preds, lr_text_probs)

    results["ablation_study"] = {
        "text_only": text_only_metrics,
        "combined": results["models"]["Logistic Regression"]["test_metrics"],
        "delta_f1": round(results["models"]["Logistic Regression"]["test_metrics"]["fraud_f1"] - text_only_metrics["fraud_f1"], 4),
        "delta_pr_auc": round(results["models"]["Logistic Regression"]["test_metrics"]["pr_auc"] - text_only_metrics["pr_auc"], 4)
    }

    # 6. Global Feature Interpretability (Top Fraud & Legit Terms)
    lr_model = classifiers["Logistic Regression"]
    top_fraud_terms, top_legit_terms = extract_top_features_from_linear_model(lr_model, preprocessor, top_n=25)
    results["global_explainability"] = {
        "top_fraud_terms": top_fraud_terms,
        "top_legit_terms": top_legit_terms
    }

    # Save metrics JSON
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll models trained and evaluated. Metrics saved to {METRICS_FILE}")
    return results


if __name__ == "__main__":
    run_training_pipeline()
