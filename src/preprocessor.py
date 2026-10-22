"""
Feature extraction and preprocessing pipelines.
Extracts TF-IDF textual features and structured metadata without leakage.
"""
import re
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from .config import (
    TEXT_COLUMNS,
    CATEGORICAL_METADATA_COLUMNS,
    BINARY_METADATA_COLUMNS,
)


def custom_text_cleaner(text: str) -> str:
    """Normalize text while retaining critical scam patterns like urls/emails."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize excessive symbols and spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text_only_preprocessor(max_features: int = 10000, ngram_range=(1, 2), min_df: int = 2):
    """TF-IDF vectorizer pipeline for text-only baseline."""
    vectorizer = TfidfVectorizer(
        preprocessor=custom_text_cleaner,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text_tfidf", vectorizer, "full_text")
        ],
        remainder="drop"
    )
    return preprocessor


def build_combined_preprocessor(max_features: int = 10000, ngram_range=(1, 2), min_df: int = 2):
    """Combined TF-IDF text features and structured metadata preprocessor."""
    vectorizer = TfidfVectorizer(
        preprocessor=custom_text_cleaner,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True
    )

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text_tfidf", vectorizer, "full_text"),
            ("cat_meta", categorical_encoder, CATEGORICAL_METADATA_COLUMNS),
            ("bin_meta", "passthrough", BINARY_METADATA_COLUMNS),
        ],
        remainder="drop"
    )
    return preprocessor
