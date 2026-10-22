"""
Inference and risk scoring engine.
Combines calibrated ML probability predictions with domain heuristic rules
and explainability signals for live screening.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.config import MODELS_DIR, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.data_loader import clean_and_feature_engineer
from src.explainability import (
    scan_heuristic_flags,
    generate_risk_assessment,
    highlight_suspicious_phrases_html
)


class RiskPredictionEngine:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.preprocessor = None
        self.models = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Load fitted ColumnTransformer and trained classifier artifacts."""
        prep_path = self.models_dir / "preprocessor.joblib"
        if prep_path.exists():
            self.preprocessor = joblib.load(prep_path)

        model_files = {
            "Logistic Regression": "logistic_regression.joblib",
            "Linear SVM": "linear_svm.joblib",
            "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
            "Random Forest": "random_forest.joblib"
        }

        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)

    @property
    def is_ready(self) -> bool:
        return self.preprocessor is not None and len(self.models) > 0

    def get_available_models(self) -> list:
        return list(self.models.keys())

    def predict(
        self,
        job_data: dict,
        model_name: str = "Linear SVM",
        low_thresh: float = LOW_RISK_THRESHOLD,
        high_thresh: float = HIGH_RISK_THRESHOLD
    ) -> dict:
        """
        Execute end-to-end risk screening on a single listing dictionary.
        Returns composite assessment with explainable tokens and heuristic flags.
        """
        if not self.is_ready:
            raise RuntimeError("Model artifacts not loaded. Ensure train_evaluate.py has been executed.")

        if model_name not in self.models:
            fallback = list(self.models.keys())[0]
            model = self.models[fallback]
            model_name = fallback
        else:
            model = self.models[model_name]

        # Normalize input row
        row = {
            "title": str(job_data.get("title", "")).strip(),
            "company_profile": str(job_data.get("company_profile", "")).strip(),
            "description": str(job_data.get("description", "")).strip(),
            "requirements": str(job_data.get("requirements", "")).strip(),
            "benefits": str(job_data.get("benefits", "")).strip(),
            "telecommuting": int(job_data.get("telecommuting", 0)),
            "has_company_logo": int(job_data.get("has_company_logo", 0)),
            "has_questions": int(job_data.get("has_questions", 0)),
            "employment_type": str(job_data.get("employment_type", "Unknown")),
            "required_experience": str(job_data.get("required_experience", "Unknown")),
            "required_education": str(job_data.get("required_education", "Unknown")),
            "salary_range": str(job_data.get("salary_range", "")).strip()
        }

        df_input = pd.DataFrame([row])
        df_clean = clean_and_feature_engineer(df_input)

        # 1. Domain Heuristic Rule Scanning
        meta_dict = {
            "has_company_logo": row["has_company_logo"],
            "has_company_profile": df_clean.iloc[0]["has_company_profile"],
            "has_questions": row["has_questions"]
        }
        heuristic_flags = scan_heuristic_flags(df_clean.iloc[0]["full_text"], meta_dict)

        # 2. Machine Learning Probabilistic Scoring
        X_trans = self.preprocessor.transform(df_clean)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)[0]
            ml_prob = float(probs[1])
            is_calibrated = True
        else:
            preds = model.predict(X_trans)[0]
            ml_prob = float(preds)
            is_calibrated = False

        # 3. Composite Assessment Synthesis
        assessment = generate_risk_assessment(ml_prob, heuristic_flags)
        
        # Override risk category if customized thresholds are passed
        if ml_prob >= high_thresh or any(f["severity"] == "HIGH" for f in heuristic_flags):
            assessment["risk_level"] = "High Risk / Fraud Alert"
            assessment["color"] = "#e53e3e"
            assessment["badge_style"] = "danger"
        elif ml_prob >= low_thresh or len(heuristic_flags) >= 2:
            assessment["risk_level"] = "Moderate / Suspicious"
            assessment["color"] = "#dd6b20"
            assessment["badge_style"] = "warning"
        else:
            assessment["risk_level"] = "Low Risk"
            assessment["color"] = "#38a169"
            assessment["badge_style"] = "success"

        # 4. Local Explanatory Markup
        assessment["highlighted_html"] = highlight_suspicious_phrases_html(df_clean.iloc[0]["full_text"])
        assessment["model_used"] = model_name
        assessment["is_calibrated"] = is_calibrated

        return assessment


# Singleton accessor
_engine_instance = None

def get_prediction_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RiskPredictionEngine()
    return _engine_instance
