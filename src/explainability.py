"""
Explainability engine providing both global feature interpretations
and local instance explanations (token highlighting, heuristic red flags, and risk scoring).
"""
import re
import numpy as np


# Common recruitment scam heuristics identified in cyber threat intelligence
SCAM_INDICATOR_PATTERNS = {
    "Financial / Wire Request": [
        r"\b(?:wire\s+transfer|western\s+union|moneygram|cashier'?s?\s+check)\b",
        r"\b(?:bitcoin|cryptocurrency|crypto\s+wallet|usdt)\b",
        r"\b(?:application\s+fee|processing\s+fee|registration\s+fee|pay\s+upfront)\b",
        r"\b(?:purchase\s+equipment|reimbursement\s+check|send\s+money)\b",
    ],
    "Suspicious Communication Channel": [
        r"\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|hotmail|outlook|live|protonmail)\.com\b",
        r"\b(?:telegram|whatsapp|signal\s+app|hangouts|skype\s+id)\b",
        r"\b(?:interview\s+via\s+(?:chat|text|telegram|whatsapp))\b",
    ],
    "Unrealistic / High Urgency Promises": [
        r"\b(?:no\s+interview\s+needed|immediate\s+hire\s+guaranteed|earn\s+\$\s*\d{3,}\s*(?:daily|weekly))\b",
        r"\b(?:envelope\s+stuffing|package\s+reshipping|mystery\s+shopper)\b",
        r"\b(?:data\s+entry\s+assistant\s+immediate|work\s+2\s+hours\s+make)\b",
    ],
    "Vague Identity / Personal Info Harvesting": [
        r"\b(?:send\s+(?:ssn|social\s+security|passport|driver'?s?\s+license|bank\s+details)\s+to\s+apply)\b",
        r"\b(?:confidential\s+interview|undisclosed\s+client)\b",
    ]
}

LEGITIMATE_SIGNALS = [
    r"\b(?:equal\s+opportunity\s+employer|eoe|affirmative\s+action)\b",
    r"\b(?:401\(?k\)?|health\s+insurance|dental|vision|paid\s+time\s+off|pto)\b",
    r"\b(?:bachelor'?s?\s+degree|master'?s?\s+degree|equivalent\s+experience)\b",
    r"\b(?:responsibilities|qualifications|years\s+of\s+experience)\b",
]


def extract_top_features_from_linear_model(model, preprocessor, top_n: int = 25):
    """
    Extract top features favoring Fraud (positive) and Legitimate (negative)
    from linear models (LogisticRegression, LinearSVC).
    """
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
    elif hasattr(model, "calibrated_classifiers_"):
        weights = [c.estimator.coef_[0] for c in model.calibrated_classifiers_ if hasattr(c.estimator, "coef_")]
        if weights:
            coef = np.mean(weights, axis=0)
        else:
            return [], []
    else:
        return [], []

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = preprocessor.named_transformers_["text_tfidf"].get_feature_names_out()

    if len(feature_names) == len(coef):
        # Clean prefix like text_tfidf__
        clean_names = [f.replace("text_tfidf__", "").replace("cat_meta__", "meta: ").replace("bin_meta__", "meta: ") for f in feature_names]
        top_fraud_indices = np.argsort(coef)[-top_n:][::-1]
        top_legit_indices = np.argsort(coef)[:top_n]

        top_fraud = [(clean_names[i], float(coef[i])) for i in top_fraud_indices]
        top_legit = [(clean_names[i], float(coef[i])) for i in top_legit_indices]
        return top_fraud, top_legit
    return [], []


def scan_heuristic_flags(text: str, metadata: dict = None) -> list:
    """Scan text and metadata against domain-specific recruitment fraud heuristics."""
    flags = []
    text_lower = text.lower() if text else ""

    # Check pattern categories
    for category, patterns in SCAM_INDICATOR_PATTERNS.items():
        matched_terms = []
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            if matches:
                matched_terms.extend(matches)
        if matched_terms:
            flags.append({
                "category": category,
                "severity": "HIGH" if "Financial" in category else "MEDIUM",
                "evidence": list(set(matched_terms))[:3],
                "description": f"Detected suspicious keywords associated with {category.lower()}."
            })

    # Check metadata heuristics
    if metadata:
        if metadata.get("has_company_logo") == 0:
            flags.append({
                "category": "Missing Identity Marker",
                "severity": "LOW",
                "evidence": ["No Company Logo Attached"],
                "description": "Legitimate employers almost always include an official corporate logo."
            })
        if metadata.get("has_company_profile") == 0:
            flags.append({
                "category": "Omitted Company Background",
                "severity": "MEDIUM",
                "evidence": ["Blank Company Profile"],
                "description": "No company history, mission, or corporate background provided."
            })
        if metadata.get("has_questions") == 0:
            flags.append({
                "category": "Zero Screening Questions",
                "severity": "LOW",
                "evidence": ["No Screening Questions"],
                "description": "Legitimate hiring workflows typically include role-specific pre-screening questions."
            })

    return flags


def generate_risk_assessment(ml_probability: float, heuristic_flags: list) -> dict:
    """
    Produce composite risk rating, clear risk tier, and human-understandable guidance.
    Tiers:
      - LOW RISK (0-24% ML prob & <= 1 low flag)
      - MODERATE / SUSPICIOUS (25-64% ML prob OR >= 2 flags)
      - HIGH RISK / FRAUD ALERT (>= 65% ML prob OR critical financial flag)
    """
    high_sev_count = sum(1 for f in heuristic_flags if f.get("severity") == "HIGH")
    total_flags = len(heuristic_flags)

    if ml_probability >= 0.65 or high_sev_count >= 1:
        risk_level = "High Risk / Fraud Alert"
        color = "#e53e3e"
        badge_style = "danger"
        summary = "This posting exhibits high statistical or heuristic indicators of recruitment fraud."
    elif ml_probability >= 0.25 or total_flags >= 2:
        risk_level = "Moderate / Suspicious"
        color = "#dd6b20"
        badge_style = "warning"
        summary = "This listing contains several ambiguous or unusual characteristics. Exercise caution."
    else:
        risk_level = "Low Risk"
        color = "#38a169"
        badge_style = "success"
        summary = "This listing closely resembles typical legitimate job advertisements."

    # Actionable safety recommendations
    recommendations = [
        "Never send money, purchase equipment, or cash checks sent by prospective employers.",
        "Verify the vacancy on the company's official corporate careers website before submitting personal details.",
        "Be wary of recruiters insisting on communicating solely through Telegram, WhatsApp, or personal email addresses.",
        "Protect personal identifiers (SSN, passport, bank account numbers) until a formal verified offer is executed."
    ]

    return {
        "risk_level": risk_level,
        "color": color,
        "badge_style": badge_style,
        "ml_probability": float(ml_probability),
        "flags_count": total_flags,
        "heuristic_flags": heuristic_flags,
        "summary": summary,
        "recommendations": recommendations,
        "disclaimer": (
            "System Limitation: This analyzer provides preliminary screening assistance based on historical "
            "patterns. It does not constitute official legal or background verification of an employer. "
            "Modern threat actors continuously adapt phrasing to evade keyword filters."
        )
    }


def highlight_suspicious_phrases_html(text: str) -> str:
    """Highlight matched keywords in HTML for explainability inspection."""
    if not text:
        return ""
    
    highlighted = text
    # Collect all regexes to highlight
    for patterns in SCAM_INDICATOR_PATTERNS.values():
        for pat in patterns:
            highlighted = re.sub(
                pat,
                r'<span style="background-color: #feb2b2; color: #9b2c2c; font-weight: bold; padding: 2px 4px; border-radius: 4px;">\g<0></span>',
                highlighted,
                flags=re.IGNORECASE
            )

    for pat in LEGITIMATE_SIGNALS:
        highlighted = re.sub(
            pat,
            r'<span style="background-color: #c6f6d5; color: #22543d; font-weight: bold; padding: 2px 4px; border-radius: 4px;">\g<0></span>',
            highlighted,
            flags=re.IGNORECASE
        )

    return highlighted
