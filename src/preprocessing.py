"""
Data preprocessing and feature extraction module.
Wraps TF-IDF vectorization and structured metadata transformations.
"""
from src.preprocessor import (
    custom_text_cleaner,
    build_text_only_preprocessor,
    build_combined_preprocessor
)

__all__ = [
    "custom_text_cleaner",
    "build_text_only_preprocessor",
    "build_combined_preprocessor"
]
