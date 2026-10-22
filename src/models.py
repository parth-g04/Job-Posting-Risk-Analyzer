"""
Model definitions for the four classifiers:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (with Platt calibration for probabilities)
4. Random Forest
All configured with class imbalance mitigation.
"""
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from .config import RANDOM_STATE


def get_classifiers(random_state: int = RANDOM_STATE) -> dict:
    """
    Returns a dictionary of uncalibrated or natively probability-capable classifiers,
    plus a Majority-Class Dummy baseline.
    LinearSVC is wrapped in CalibratedClassifierCV for probability output.
    """
    linear_svc = LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state
    )
    calibrated_svc = CalibratedClassifierCV(
        estimator=linear_svc,
        method="sigmoid",
        cv=3
    )

    models = {
        "Majority Class Baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.1
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state
        ),
        "Linear SVM": calibrated_svc,
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=35,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
    }

    return models
