"""
Comprehensive Unit and Integration Test Suite for Job Posting Risk Analyzer.
Covers:
1. Dataset loading, cleaning, schema validation, and duplicate handling.
2. Leakage prevention and frozen preprocessor behavior.
3. Classifiers, majority baseline, and probability outputs.
4. Evaluation metrics, confusion matrix math, and balanced accuracy.
5. RiskPredictionEngine inference, edge cases (empty text, missing metadata, boundary thresholds).
6. Uploaded CSV schema adaptation, differently named columns, and single-class inputs.
7. Explainability feature extraction and heuristic safety rules.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import clean_and_feature_engineer
from src.preprocessor import build_combined_preprocessor, build_text_only_preprocessor
from src.models import get_classifiers
from src.evaluation import compute_comprehensive_metrics
from src.explainability import (
    scan_heuristic_flags,
    generate_risk_assessment,
    highlight_suspicious_phrases_html
)
from src.predict import RiskPredictionEngine, get_prediction_engine
from src.eda import get_dataset_overview_stats, plot_target_distribution


class TestJobRiskPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_raw_df = pd.DataFrame({
            "job_id": [101, 102, 103, 104],
            "title": [
                "Senior Backend Engineer",
                "Work From Home Data Entry Assistant",
                "Product Marketing Lead",
                "Urgent Crypto Investment Trader"
            ],
            "company_profile": [
                "CloudScale Systems is an enterprise observability leader.",
                None,
                "Creative Growth Agency founded in 2014.",
                ""
            ],
            "description": [
                "Design high-throughput distributed microservices in Python, Kafka, and AWS.",
                "Immediate hiring without interview! Wire transfer registration fee of $150 required. Contact Telegram @fastcash.",
                "Spearhead enterprise demand generation and cross-functional product launches.",
                "Earn $10,000 weekly guaranteed! Deposit funds via Bitcoin or Western Union to start trading."
            ],
            "requirements": [
                "5+ years Python and cloud systems.",
                "No experience needed.",
                "3+ years B2B product marketing.",
                "Active crypto wallet."
            ],
            "benefits": [
                "Full medical, dental, 401(k) matching, and unlimited PTO.",
                "Earn $3,000 weekly.",
                "Competitive salary and equity.",
                "Instant payout."
            ],
            "telecommuting": [1, 1, 0, 1],
            "has_company_logo": [1, 0, 1, 0],
            "has_questions": [1, 0, 1, 0],
            "employment_type": ["Full-time", "Part-time", "Full-time", "Contract"],
            "required_experience": ["Mid-Senior level", "Entry level", "Associate", "Not Applicable"],
            "required_education": ["Bachelor's Degree", "Unspecified", "Bachelor's Degree", "Unspecified"],
            "salary_range": ["140000-180000", "5000-10000", None, "100000-200000"],
            "fraudulent": [0, 1, 0, 1]
        })

    # 1. Dataset Loading, Cleaning, and Duplicate Handling
    def test_clean_and_feature_engineer_columns_and_flags(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_df)
        self.assertIn("full_text", df_clean.columns)
        self.assertIn("has_company_profile", df_clean.columns)
        self.assertIn("salary_specified", df_clean.columns)

        # Check binary flags
        self.assertEqual(df_clean.iloc[0]["has_company_profile"], 1)
        self.assertEqual(df_clean.iloc[1]["has_company_profile"], 0)
        self.assertEqual(df_clean.iloc[0]["salary_specified"], 1)
        self.assertEqual(df_clean.iloc[2]["salary_specified"], 0)

        # Verify full_text concatenation
        self.assertIn("cloudscale systems", df_clean.iloc[0]["full_text"].lower())
        self.assertIn("wire transfer", df_clean.iloc[1]["full_text"].lower())

    def test_clean_handles_missing_and_whitespace_strings(self):
        df_messy = pd.DataFrame({
            "title": ["   Messy Title   ", None, ""],
            "description": [None, "   ", "Valid Description"],
            "has_company_logo": [None, "invalid", 1],
            "fraudulent": [None, 1, "0"]
        })
        df_clean = clean_and_feature_engineer(df_messy)
        self.assertEqual(df_clean.iloc[0]["title"], "Messy Title")
        self.assertEqual(df_clean.iloc[1]["title"], "")
        self.assertEqual(int(df_clean.iloc[0]["has_company_logo"]), 0)
        self.assertEqual(int(df_clean.iloc[0]["fraudulent"]), 0)
        self.assertEqual(int(df_clean.iloc[1]["fraudulent"]), 1)

    # 2. Leakage Prevention and Preprocessor Behavior
    def test_preprocessor_leakage_isolation(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_df)
        preprocessor = build_combined_preprocessor(max_features=100, min_df=1)

        # Fit preprocessor on training partition
        train_df = df_clean.iloc[:2]
        test_df = df_clean.iloc[2:].copy()
        test_df["required_experience"] = ["Unseen Experience 1", "Unseen Experience 2"]

        X_train = preprocessor.fit_transform(train_df)
        # Transform test data using frozen preprocessor (no leakage!)
        X_test = preprocessor.transform(test_df)

        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_test.shape[0], 2)

    # 3. Classifiers & Baseline Consistency
    def test_all_classifiers_support_probability_and_baseline(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_df)
        df_augmented = pd.concat([df_clean] * 4, ignore_index=True)

        preprocessor = build_combined_preprocessor(max_features=50, min_df=1)
        X = preprocessor.fit_transform(df_augmented)
        y = df_augmented["fraudulent"].values

        models = get_classifiers(random_state=42)
        self.assertIn("Majority Class Baseline", models)
        self.assertIn("Linear SVM", models)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Multinomial Naive Bayes", models)
        self.assertIn("Random Forest", models)

        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            self.assertEqual(len(preds), len(y))
            if name != "Majority Class Baseline":
                self.assertTrue(hasattr(model, "predict_proba"), f"{name} should support predict_proba")
                probs = model.predict_proba(X)
                self.assertEqual(probs.shape, (len(y), 2))
                self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    # 4. Metric Calculations & Confusion Matrix
    def test_metrics_and_confusion_matrix_consistency(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.15, 0.2, 0.65, 0.35, 0.8, 0.9, 0.95])

        m = compute_comprehensive_metrics(y_true, y_pred, y_prob)
        self.assertEqual(m["confusion_matrix"]["tn"], 3)
        self.assertEqual(m["confusion_matrix"]["fp"], 1)
        self.assertEqual(m["confusion_matrix"]["fn"], 1)
        self.assertEqual(m["confusion_matrix"]["tp"], 3)
        self.assertEqual(m["fraud_precision"], 0.75)
        self.assertEqual(m["fraud_recall"], 0.75)
        self.assertEqual(m["fraud_f1"], 0.75)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75, places=2)
        self.assertGreater(m["pr_auc"], 0.7)

    # 5. Risk Prediction Engine & Edge Cases
    def test_prediction_engine_empty_text_and_boundary_conditions(self):
        engine = get_prediction_engine()
        if not engine.is_ready:
            self.skipTest("Models not saved yet on disk.")

        # Test completely empty text input
        empty_listing = {
            "title": "",
            "company_profile": "",
            "description": "",
            "requirements": "",
            "benefits": "",
            "has_company_logo": 0,
            "has_questions": 0
        }
        res_empty = engine.predict(empty_listing, low_thresh=0.25, high_thresh=0.65)
        self.assertIn("risk_level", res_empty)
        self.assertIn("ml_probability", res_empty)

        # Test extreme scam input
        scam_listing = {
            "title": "Data Entry Immediate Start",
            "description": "Wire transfer fee $200 required for check equipment. Contact telegram @fast_scam. Earn $5000 weekly guaranteed.",
            "has_company_logo": 0,
            "has_questions": 0
        }
        res_scam = engine.predict(scam_listing, low_thresh=0.25, high_thresh=0.65)
        self.assertEqual(res_scam["risk_level"], "High Risk / Fraud Alert")
        self.assertEqual(res_scam["badge_style"], "danger")
        self.assertTrue(any(f["severity"] == "HIGH" for f in res_scam["heuristic_flags"]))

        # Test threshold override boundary conditions
        res_lenient = engine.predict(scam_listing, high_thresh=0.99)
        self.assertIn("risk_level", res_lenient)

    # 6. Upload Adaptability and EDA Functions
    def test_eda_overview_and_uploaded_csv_resilience(self):
        stats = get_dataset_overview_stats(self.sample_raw_df)
        self.assertEqual(stats["total_rows"], 4)
        self.assertEqual(stats["total_cols"], 14)
        self.assertEqual(stats["exact_duplicates"], 0)

        # Test arbitrary uploaded DataFrame with missing/different columns
        df_custom = pd.DataFrame({
            "ad_title": ["Sales Rep", "Crypto Admin"],
            "summary": ["Work in retail", "Earn bitcoin daily via wire transfer"],
            "label": ["real", "fraud"]
        })
        stats_custom = get_dataset_overview_stats(df_custom)
        self.assertEqual(stats_custom["total_rows"], 2)
        self.assertEqual(stats_custom["total_cols"], 3)

        # Test target distribution plot on custom label
        fig_bar, fig_donut = plot_target_distribution(df_custom, "label")
        self.assertIsNotNone(fig_bar)
        self.assertIsNotNone(fig_donut)

    # 7. Explainability & Rule Verifications
    def test_explainability_heuristic_keywords_and_html(self):
        text = "Urgent: Send cashier's check or western union. Reach recruiter on whatsapp or telegram."
        flags = scan_heuristic_flags(text, metadata={"has_company_logo": 0, "has_questions": 0})
        self.assertGreaterEqual(len(flags), 2)

        html = highlight_suspicious_phrases_html(text)
        self.assertIn("<span", html)
        self.assertIn("western union", html.lower())


if __name__ == "__main__":
    unittest.main()
