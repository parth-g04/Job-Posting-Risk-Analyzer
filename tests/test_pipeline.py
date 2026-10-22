"""
Unit and integration tests for the Explainable Job Posting Risk Analyzer pipeline.
Uses standard Python unittest.
"""
import unittest
import numpy as np
import pandas as pd
from src.data_loader import clean_and_feature_engineer
from src.preprocessor import build_combined_preprocessor
from src.models import get_classifiers
from src.evaluation import compute_comprehensive_metrics
from src.explainability import (
    scan_heuristic_flags,
    generate_risk_assessment,
    highlight_suspicious_phrases_html
)


class TestJobRiskPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_raw_dataframe = pd.DataFrame({
            "job_id": [1, 2, 3],
            "title": ["Senior Software Engineer", "Data Entry Clerk Urgently Needed", "Marketing Lead"],
            "company_profile": ["TechCorp is a leading innovator in cloud observability...", None, "Creative agency founded in 2010."],
            "description": [
                "We are looking for a Python expert with AWS experience.",
                "Work from home guaranteed! Wire transfer fee required for home office equipment. Contact on telegram @scam_job.",
                "Manage organic growth and lead generation campaigns."
            ],
            "requirements": ["5+ years Python, Docker, Kubernetes.", "No experience required, immediate start.", "3+ years in B2B marketing."],
            "benefits": ["Full healthcare, 401(k) matching, flexible PTO.", "Earn $5000 weekly without interview.", "Health insurance and bonus."],
            "telecommuting": [1, 1, 0],
            "has_company_logo": [1, 0, 1],
            "has_questions": [1, 0, 1],
            "employment_type": ["Full-time", "Part-time", "Full-time"],
            "required_experience": ["Mid-Senior level", "Entry level", "Associate"],
            "required_education": ["Bachelor's Degree", "Unspecified", "Bachelor's Degree"],
            "salary_range": ["120000-150000", "5000-10000", None],
            "fraudulent": [0, 1, 0]
        })

    def test_clean_and_feature_engineer(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_dataframe)
        self.assertIn("full_text", df_clean.columns)
        self.assertIn("has_company_profile", df_clean.columns)
        self.assertIn("salary_specified", df_clean.columns)

        # Row 0: legitimate, Row 1: fraud
        self.assertEqual(df_clean.iloc[0]["has_company_profile"], 1)
        self.assertEqual(df_clean.iloc[1]["has_company_profile"], 0)
        self.assertEqual(df_clean.iloc[0]["has_company_logo"], 1)
        self.assertEqual(df_clean.iloc[1]["has_company_logo"], 0)
        self.assertIn("python", df_clean.iloc[0]["full_text"].lower())
        self.assertIn("wire transfer", df_clean.iloc[1]["full_text"].lower())

    def test_preprocessor_transformation(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_dataframe)
        preprocessor = build_combined_preprocessor(max_features=100, min_df=1)

        X = preprocessor.fit_transform(df_clean)
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)

        # Test transformation on unseen data
        df_unseen = df_clean.copy()
        df_unseen["required_experience"] = ["Novel Experience", "Unknown", "Expert"]
        X_unseen = preprocessor.transform(df_unseen)
        self.assertEqual(X_unseen.shape[0], 3)
        self.assertEqual(X_unseen.shape[1], X.shape[1])

    def test_classifiers_and_prob_outputs(self):
        df_clean = clean_and_feature_engineer(self.sample_raw_dataframe)
        # Duplicate to have enough samples for k-fold probability calibration
        df_augmented = pd.concat([df_clean] * 4, ignore_index=True)

        preprocessor = build_combined_preprocessor(max_features=50)
        X = preprocessor.fit_transform(df_augmented)
        y = df_augmented["fraudulent"].values

        models = get_classifiers(random_state=42)
        self.assertEqual(len(models), 4)

        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            self.assertEqual(len(preds), len(y))

            # Verify probability outputs
            self.assertTrue(hasattr(model, "predict_proba"), f"{name} must support predict_proba")
            probs = model.predict_proba(X)
            self.assertEqual(probs.shape, (len(y), 2))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_evaluation_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.85])

        metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)
        self.assertIn("fraud_precision", metrics)
        self.assertIn("fraud_recall", metrics)
        self.assertIn("fraud_f1", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)

    def test_explainability_heuristics(self):
        scam_text = "Earn $5000 weekly! Wire transfer required for background fee. Contact on telegram @recruiter."
        metadata = {
            "has_company_logo": 0,
            "has_company_profile": 0,
            "has_questions": 0
        }

        flags = scan_heuristic_flags(scam_text, metadata)
        self.assertGreaterEqual(len(flags), 2)

        # Verify risk assessment
        assessment = generate_risk_assessment(ml_probability=0.88, heuristic_flags=flags)
        self.assertEqual(assessment["risk_level"], "High Risk / Fraud Alert")
        self.assertEqual(assessment["badge_style"], "danger")
        self.assertGreater(len(assessment["recommendations"]), 0)

        # Test HTML highlight
        highlighted = highlight_suspicious_phrases_html(scam_text)
        self.assertIn("<span", highlighted)
        self.assertIn("wire transfer", highlighted.lower())


if __name__ == "__main__":
    unittest.main()
