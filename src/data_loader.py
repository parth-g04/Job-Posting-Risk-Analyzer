"""
Data loader and splitting pipeline for Job Posting Risk Analyzer.
Implements leak-free stratified dataset preparation.
"""
import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import (
    RAW_DATA_FILE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    DATASET_URL,
    TEXT_COLUMNS,
    CATEGORICAL_METADATA_COLUMNS,
    BINARY_METADATA_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def download_raw_dataset(dest_file=RAW_DATA_FILE, force=False) -> str:
    """Download the raw dataset if not already present."""
    dest_file = dest_file or RAW_DATA_FILE
    dest_dir = dest_file.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_file.exists() and not force:
        if dest_file.stat().st_size > 1024 * 1024:  # At least 1MB
            return str(dest_file)

    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, dest_file)
    print(f"Downloaded raw dataset to {dest_file} ({dest_file.stat().st_size / (1024 * 1024):.2f} MB)")
    return str(dest_file)


def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and construct structured features alongside text concatenation."""
    df = df.copy()

    # Ensure target column is integer binary
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

    # Missing value handling for text columns
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Engineered binary metadata
    df["has_company_profile"] = (df["company_profile"].str.len() > 10).astype(int)
    
    if "salary_range" in df.columns:
        df["salary_specified"] = df["salary_range"].fillna("").astype(str).str.strip().apply(
            lambda x: 1 if len(x) > 0 and x.lower() != "nan" else 0
        )
    else:
        df["salary_specified"] = 0

    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    # Categorical metadata cleaning
    for col in CATEGORICAL_METADATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        else:
            df[col] = "Unknown"

    # Combined full text representation for TF-IDF
    df["full_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def prepare_and_split_data(raw_file=RAW_DATA_FILE, force_download=False):
    """
    Load raw data, clean, create stratified train/val/test splits, and save.
    Stratification ensures balanced positive class ratios across splits.
    """
    download_raw_dataset(dest_file=raw_file, force=force_download)
    
    print(f"Loading raw data from {raw_file}...")
    df = pd.read_csv(raw_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    df_clean = clean_and_feature_engineer(df)
    
    y = df_clean[TARGET_COLUMN]
    
    # 70% train, 30% temp (which will be split 50/50 into 15% val and 15% test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    df_train, df_temp = train_test_split(
        df_clean,
        test_size=val_test_ratio,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split the temp set equally into Val and Test
    test_share_of_temp = TEST_RATIO / val_test_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_share_of_temp,
        random_state=RANDOM_STATE,
        stratify=df_temp[TARGET_COLUMN]
    )

    # Save to processed directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_DATA_FILE, index=False)
    df_val.to_csv(VAL_DATA_FILE, index=False)
    df_test.to_csv(TEST_DATA_FILE, index=False)

    print(f"Saved splits to {PROCESSED_DATA_DIR}:")
    print(f"  Train: {len(df_train)} rows (Fraudulent: {df_train[TARGET_COLUMN].sum()} - {df_train[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Val:   {len(df_val)} rows (Fraudulent: {df_val[TARGET_COLUMN].sum()} - {df_val[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"  Test:  {len(df_test)} rows (Fraudulent: {df_test[TARGET_COLUMN].sum()} - {df_test[TARGET_COLUMN].mean()*100:.2f}%)")

    return df_train, df_val, df_test


if __name__ == "__main__":
    prepare_and_split_data()
