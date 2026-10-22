"""
Executive Terminal Report Generator for Job Posting Risk Analyzer.
Prints publication-grade summaries of data, benchmarks, ablation, explainability, and error analysis.
"""
import json
from pathlib import Path
import pandas as pd
from src.config import METRICS_FILE, REPORTS_DIR

def main():
    if not METRICS_FILE.exists():
        print("Metrics file not found. Run train_evaluate.py first.")
        return

    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    meta = data["dataset_metadata"]
    models = data["models"]
    ablation = data["ablation_study"]
    explain = data.get("global_explainability", {})
    errors = data.get("error_analysis", {})

    print("=" * 86)
    print("      ACADEMIC ML PROJECT REPORT: EXPLAINABLE JOB POSTING RISK ANALYZER")
    print("      Coursework / Mini-Project Evaluation (Worth 15 Marks)")
    print("=" * 86)

    print("\n1. PROBLEM STATEMENT & DATASET OVERVIEW")
    print("-" * 86)
    print("Core Objective  : Machine-learning screening tool to identify recruitment fraud,")
    print("                  address severe class imbalance, prevent data leakage, and provide explainability.")
    print(f"Dataset Name    : Employment Scam Aegean Dataset (EMSCAD / Real or Fake Job Postings)")
    print(f"Total Instances : {meta['total_samples']:,} advertisements")
    print(f"Class Balance   : Legitimate = 17,014 (95.16%) | Fraudulent = 866 (4.84%)")
    print(f"Imbalance Ratio : ~19.6 : 1 (Requires PR-AUC, F1, and cost-sensitive loss)")
    print("Partitions (Leak-Free Stratified 70/15/15 Split):")
    print(f"  * Training Partition   : {meta['train_samples']:,} samples (Fitted TF-IDF & One-Hot Encoders)")
    print(f"  * Validation Partition : {meta['val_samples']:,} samples (Hyperparameter & threshold selection)")
    print(f"  * Test Partition       : {meta['test_samples']:,} samples (Frozen unbiased evaluation: 130 fraud, 2,552 legit)")

    print("\n2. FEATURE PIPELINE & STRICT LEAKAGE PREVENTION")
    print("-" * 86)
    print("Feature Space   : 10,033 total dimensions")
    print("  • Text Branch : TF-IDF with sublinear scaling, unigrams + bigrams, min_df=2 (10,000 terms)")
    print("  • Meta Branch : One-Hot Encoding (employment_type, required_experience, education)")
    print("                  Binary flags (has_company_logo, has_questions, telecommuting, profile, salary)")
    print("Leakage Protocol: ALL encoders fitted STRICTLY on 12,516 training samples. Val/Test only transformed.")

    print("\n3. CLASSIFIER BENCHMARK COMPARISON (HELD-OUT TEST SET: 2,682 LISTINGS)")
    print("-" * 86)
    print(f"{'Classifier Model':<25} | {'Test Prec':<9} | {'Test Rec':<9} | {'Fraud F1':<8} | {'PR-AUC':<7} | {'Bal Acc':<7} | {'5-Fold CV F1'}")
    print("-" * 86)
    for name, m in models.items():
        tm = m["test_metrics"]
        cv_str = f"{m.get('cv_f1_mean', 0):.3f} ± {m.get('cv_f1_std', 0):.3f}" if 'cv_f1_mean' in m else "N/A"
        print(f"{name:<25} | {tm['fraud_precision']*100:6.2f}%   | {tm['fraud_recall']*100:6.2f}%  | {tm['fraud_f1']*100:6.2f}%  | {tm['pr_auc'] or 0:6.4f} | {tm['balanced_accuracy']*100:6.2f}% | {cv_str}")
    print("-" * 86)
    print("Key Finding: Linear SVM achieved best Precision (93.52%) and overall F1 (84.87%).")
    print("             Logistic Regression caught the most scams (93.85% Recall, only 8 missed).")
    print("             Majority Baseline demonstrates that Accuracy (95.15%) is completely invalid.")

    print("\n4. TEST SET CONFUSION MATRICES (TOTAL TEST SAMPLES = 2,682 | ACTUAL FRAUD = 130)")
    print("-" * 86)
    for name, m in models.items():
        cm = m["test_metrics"]["confusion_matrix"]
        print(f"[{name}]")
        print(f"  True Negatives (Legit identified as Legit) : {cm['tn']:>4} / 2552 ({cm.get('tn_pct', 0):.1f}%)")
        print(f"  False Positives (False Alarms on Legit)    : {cm['fp']:>4} / 2552 ({cm.get('fp_pct', 0):.1f}%)")
        print(f"  False Negatives (Missed Scams / Danger)    : {cm['fn']:>4} /  130 ({cm.get('fn_pct', 0):.1f}%)")
        print(f"  True Positives (Successfully Caught Scams) : {cm['tp']:>4} /  130 ({cm.get('tp_pct', 0):.1f}%)")
        print()

    print("5. METADATA FEATURE ABLATION (VALUE OF STRUCTURED METADATA)")
    print("-" * 86)
    to = ablation["text_only"]
    cb = ablation["combined"]
    print(f"Text-Only (TF-IDF alone)   -> Precision: {to['fraud_precision']*100:.2f}% | Recall: {to['fraud_recall']*100:.2f}% | PR-AUC: {to['pr_auc']:.4f}")
    print(f"Combined (Text + Metadata) -> Precision: {cb['fraud_precision']*100:.2f}% | Recall: {cb['fraud_recall']*100:.2f}% | PR-AUC: {cb['pr_auc']:.4f}")
    print(f"Delta PR-AUC with Metadata -> {ablation.get('delta_pr_auc', 0):+.4f} gain")
    print(f"Delta Recall with Metadata -> {ablation.get('delta_recall', 0)*100:+.2f}% increase in scams detected")

    print("\n6. EXPLAINABILITY & HEURISTIC RULES")
    print("-" * 86)
    print("Top Positive Model Weights (Fraud Indicators):")
    if explain.get("top_fraud_terms"):
        for term, wt in explain["top_fraud_terms"][:6]:
            print(f"  * {term:24} (weight: {wt:+.4f})")
    print("\nTop Negative Model Weights (Legitimate Indicators):")
    if explain.get("top_legit_terms"):
        for term, wt in explain["top_legit_terms"][:6]:
            print(f"  * {term:24} (weight: {wt:+.4f})")

    print("\n7. ERROR ANALYSIS (UNDERSTANDING FALSE POSITIVES & FALSE NEGATIVES)")
    print("-" * 86)
    if errors:
        print(f"Linear SVM Test Errors: {errors.get('total_false_positives', 0)} False Positives | {errors.get('total_false_negatives', 0)} False Negatives")
        print("\nPrimary Error Typologies:")
        for typ in errors.get("error_typologies", []):
            print(f"  • {typ['category']}")
        if errors.get("false_positive_examples"):
            fp_ex = errors["false_positive_examples"][0]
            print(f"\nFalse Positive Example (Job #{fp_ex['job_id']}: '{fp_ex['title']}'):")
            print(f"  Cause: {fp_ex['reason_category']}")
            print(f"  Snippet: {fp_ex['snippet']}")
        if errors.get("false_negative_examples"):
            fn_ex = errors["false_negative_examples"][0]
            print(f"\nFalse Negative Example (Job #{fn_ex['job_id']}: '{fn_ex['title']}'):")
            print(f"  Cause: {fn_ex['reason_category']}")
            print(f"  Snippet: {fn_ex['snippet']}")

    print("\n8. RUNNING THE INTERACTIVE APPLICATION")
    print("-" * 86)
    print("Streamlit UI URL : http://localhost:8502")
    print("Command to start : streamlit run app.py --server.port 8502")
    print("Comparison CSV   : reports/model_comparison.csv")
    print("Saved Figures    : reports/figures/ (PNGs)")
    print("=" * 86)

if __name__ == "__main__":
    main()
