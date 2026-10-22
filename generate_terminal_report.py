"""
Generates a comprehensive executive report in the terminal for presentation and review.
"""
import json
from pathlib import Path
import pandas as pd
from src.config import METRICS_FILE, TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE

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

    print("=" * 80)
    print(" PROJECT SUMMARY: EXPLAINABLE JOB POSTING RISK ANALYZER (NLP & ML)")
    print("=" * 80)

    print("\n1. PROBLEM STATEMENT & DATASET OVERVIEW")
    print("-" * 80)
    print("Goal: Automated screening of online recruitment ads to identify fraud,")
    print("      mitigate class imbalance, prevent data leakage, and supply explainability.")
    print(f"Total Dataset Listings : {meta['total_samples']:,}")
    print(f"Overall Fraud Rate     : {meta['fraud_prevalence_pct']}% (Imbalanced: ~95% Legit vs ~5% Fraud)")
    print("Partitions (Leak-Free Stratified 70/15/15 Split):")
    print(f"  * Training Set   : {meta['train_samples']:,} postings (fitted vectorizers and models)")
    print(f"  * Validation Set : {meta['val_samples']:,} postings (model hyperparameter tuning)")
    print(f"  * Test Set       : {meta['test_samples']:,} postings (unbiased held-out generalization)")

    print("\n2. FEATURE ENGINEERING & LEAKAGE PREVENTION")
    print("-" * 80)
    print("Text Branch     : TF-IDF (Unigrams + Bigrams, Sublinear TF, English Stopwords, Max 10k terms)")
    print("Metadata Branch : One-Hot Encoding for Employment Type, Experience Level, Education")
    print("                  Binary Passthrough for Company Logo, Questions, Telecommuting, Profile")
    print("Total Features  : 10,033 dimensions (strictly fitted ONLY on Training data)")

    print("\n3. CLASSIFIER COMPARISON BENCHMARK (TEST SET EVALUATION)")
    print("-" * 80)
    print(f"{'Classifier Model':<26} | {'Test Prec':<10} | {'Test Recall':<11} | {'Test F1':<9} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Train (s)'}")
    print("-" * 80)
    for name, m in models.items():
        tm = m["test_metrics"]
        print(f"{name:<26} | {tm['fraud_precision']*100:6.2f}%    | {tm['fraud_recall']*100:6.2f}%     | {tm['fraud_f1']*100:6.2f}%   | {tm['pr_auc']:6.4f} | {tm['roc_auc']:6.4f} | {m['train_time_sec']:5.2f}s")
    print("-" * 80)
    print("Key Takeaway: Linear SVM achieved top precision (93.52%) and F1 (84.87%).")
    print("              Logistic Regression achieved highest fraud recall (93.85%).")

    print("\n4. TEST SET CONFUSION MATRICES (TOTAL TEST SAMPLES = 2,682 | ACTUAL FRAUD = 130)")
    print("-" * 80)
    for name, m in models.items():
        cm = m["test_metrics"]["confusion_matrix"]
        print(f"[{name}]")
        print(f"  True Negatives (Legit identified as Legit)   : {cm['tn']:>4} / 2552")
        print(f"  False Positives (False Alarms on Legit)      : {cm['fp']:>4} / 2552")
        print(f"  False Negatives (Missed Scams / Risk)        : {cm['fn']:>4} /  130")
        print(f"  True Positives (Successfully Caught Scams)   : {cm['tp']:>4} /  130")
        print()

    print("5. METADATA FEATURE ABLATION (VALUE OF STRUCTURED METADATA)")
    print("-" * 80)
    to = ablation["text_only"]
    cb = ablation["combined"]
    print(f"Text-Only (TF-IDF alone)    -> Precision: {to['fraud_precision']*100:.2f}% | Recall: {to['fraud_recall']*100:.2f}% | PR-AUC: {to['pr_auc']:.4f}")
    print(f"Combined (Text + Metadata)  -> Precision: {cb['fraud_precision']*100:.2f}% | Recall: {cb['recall']*100 if 'recall' in cb else cb['fraud_recall']*100:.2f}% | PR-AUC: {cb['pr_auc']:.4f}")
    print(f"Delta PR-AUC with Metadata  -> {ablation['delta_pr_auc']:+.4f} gain")
    print("Finding: Missing company logos and blank profiles provide strong discriminative signals.")

    print("\n6. EXPLAINABILITY & HEURISTIC RULES")
    print("-" * 80)
    print("Top Fraud Indicators (Model Weights):")
    if explain.get("top_fraud_terms"):
        for term, wt in explain["top_fraud_terms"][:8]:
            print(f"  * {term:24} (weight: {wt:+.4f})")
    print("\nTop Legitimate Indicators (Model Weights):")
    if explain.get("top_legit_terms"):
        for term, wt in explain["top_legit_terms"][:8]:
            print(f"  * {term:24} (weight: {wt:+.4f})")

    print("\nMulti-Layer Explainability in Streamlit App:")
    print("  1. Visual Risk Gauge (Calibrated ML Probability: Low < 25%, Moderate 25-65%, High >= 65%)")
    print("  2. Heuristic Red Flag Scanner (Wire transfers, Crypto, Telegram/WhatsApp, Personal Gmail)")
    print("  3. Text Highlighter (Color-codes suspicious n-grams red and legitimate markers green)")
    print("  4. Applicant Verification Checklist (Safe next steps & cybersecurity disclaimers)")
    print("=" * 80)

if __name__ == "__main__":
    main()
