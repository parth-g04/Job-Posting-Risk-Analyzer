"""
Configuration and constants for Job Posting Risk Analyzer.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_FILE = RAW_DATA_DIR / "fake_job_postings.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
METRICS_FILE = MODELS_DIR / "evaluation_results.json"

# Dataset URL (Reliable GitHub mirror of the Kaggle/EMSCAD dataset)
DATASET_URL = "https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv"

# Text and Metadata Columns
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

CATEGORICAL_METADATA_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education"
]

BINARY_METADATA_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "has_company_profile",
    "salary_specified"
]

TARGET_COLUMN = "fraudulent"

# Splits
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Risk Thresholds
LOW_RISK_THRESHOLD = 0.25
HIGH_RISK_THRESHOLD = 0.65
