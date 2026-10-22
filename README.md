# Explainable Job Posting Risk Analyzer Using NLP and Machine Learning

A machine-learning-based screening system that analyzes online job-posting text and structured metadata to identify patterns reminiscent of fraudulent job advertisements. The system pairs statistical model predictions with human-interpretable warning signals, domain heuristics, and explicit limitations.

---

## 🎯 Problem Statement & Overview

Online recruitment fraud exposes job seekers to identity theft, financial loss, and credential harvesting. Manual inspection of listings across aggregators is labor-intensive and prone to fatigue. 

This project implements a multi-tier screening system:
1. **Natural Language Processing (NLP)**: TF-IDF feature extraction with sublinear term frequencies, n-gram ranges, and stopword filtering.
2. **Structured Metadata Features**: Evaluates company logo presence, screening questions, telecommuting indicators, company profiles, and employment metadata.
3. **Imbalanced Machine Learning**: Compares four distinct classification algorithms:
   - **Multinomial Naive Bayes** (fast probabilistic baseline)
   - **Logistic Regression** (calibrated baseline with interpretable log-odds coefficients)
   - **Linear Support Vector Machine (Linear SVM)** (maximal margin classification with Platt scaling)
   - **Random Forest** (ensemble modeling non-linear interactions)
4. **Explainability & Risk Categorization**:
   - Transparent risk tiers: **Low Risk**, **Moderate / Suspicious**, and **High Risk / Fraud Alert**.
   - Model feature weight inspection for top fraudulent and legitimate terms.
   - Text highlighter spotlighting suspicious phrases and legitimate markers.
   - Domain-specific heuristic rule scanning (unprofessional email providers, off-platform communication like Telegram/WhatsApp, upfront equipment payment demands).
5. **Interactive Streamlit Application**:
   - Live Risk Scanner with pre-loaded sample postings.
   - Algorithmic Benchmark comparison dashboard (PR-AUC, ROC-AUC, Precision, Recall, F1-Score).
   - Metadata Feature Ablation study.
   - Dataset Exploratory Data Analysis (EDA).
   - Explicit limitations and job-seeker safety checklists.

---

## 🔬 Leakage Prevention & Imbalanced Evaluation Protocol

### 1. Zero Data Leakage
- **Split Isolation**: The raw dataset (~17,880 listings) is partitioned into **Stratified Train (70%)**, **Validation (15%)**, and **Test (15%)** subsets using a fixed random seed (`42`).
- **Feature Pipeline Isolation**: All vectorizers (`TfidfVectorizer`), categorical encoders (`OneHotEncoder`), and vocabulary indexing are fitted **strictly on the training set**. Validation and Test sets are transformed using the pre-fitted pipeline.
- **Identifier Exclusion**: Unique identifiers (`job_id`) and target columns are isolated from the input space.

### 2. Imbalanced Metric Focus
Recruitment fraud datasets exhibit severe class imbalance (~4.8% fraudulent listings). Evaluating models purely on accuracy is misleading (a naive classifier predicting all listings as legitimate achieves ~95.2% accuracy while detecting 0 scams). 
Consequently, this project evaluates:
- **Fraud Precision**: Proportion of flagged listings that are genuinely fraudulent.
- **Fraud Recall**: Proportion of actual fraudulent listings correctly detected.
- **Fraud F1-Score**: Harmonic mean of fraud precision and recall.
- **Precision-Recall AUC (PR-AUC / Average Precision)**: Primary benchmark metric reflecting performance across all operating thresholds.
- **Confusion Matrix Analysis**: Explicit tracking of False Negatives (missed scams) vs False Positives (false alarms).

---

## 🚀 Setup & Execution

### 1. Environment Setup (Windows PowerShell)

```powershell
# Create dedicated virtual environment
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Train Models & Run Benchmark Pipeline

Execute the end-to-end training and evaluation script:

```powershell
python train_evaluate.py
```

This script will:
1. Automatically download the EMSCAD/Kaggle recruitment fraud dataset (if not already downloaded).
2. Clean text, construct metadata flags, and generate leak-free stratified Train/Val/Test splits in `data/processed/`.
3. Fit the unified preprocessor and train all 4 classifiers.
4. Run the feature ablation study (Text-Only vs Text+Metadata).
5. Extract global feature importances and save all models and `models/evaluation_results.json`.

### 3. Run Interactive Streamlit App

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 4. Run Automated Unit Tests

```powershell
pytest tests/ -v
```

---

## 📁 Project Structure

```
Job Posting Risk Analyzer/
├── .venv/                      # Python virtual environment
├── data/
│   ├── raw/                    # Raw fake_job_postings.csv
│   └── processed/              # Stratified train.csv, val.csv, test.csv
├── models/
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (TF-IDF + Metadata)
│   ├── multinomial_naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── random_forest.joblib
│   └── evaluation_results.json # Full benchmark metrics & curve points
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants, column definitions, split ratios
│   ├── data_loader.py          # Data ingestion, cleaning, stratified splitting
│   ├── preprocessor.py         # TF-IDF and ColumnTransformer pipelines
│   ├── models.py               # Classifier factory configurations
│   ├── evaluation.py           # Imbalanced metrics, PR curves, ROC curves
│   └── explainability.py       # Heuristics, risk tiers, token highlighting
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Comprehensive unit & integration tests
├── app.py                      # Interactive Streamlit application
├── train_evaluate.py           # Reproducible training & evaluation CLI
├── requirements.txt
└── README.md
```

---

## ⚠️ Important Limitations

1. **Preliminary Screening Aid**: This tool assists initial triage and does not verify employer legitimacy or authenticate legal contracts.
2. **Historical Data Drift**: Scammers evolve tactics over time. While historical scams often centered on cashier's checks, wire transfers, and typo-heavy text, modern scams may use generative AI and spear-phishing.
3. **Evasion Risks**: Sophisticated adversaries can copy authentic job descriptions and alter only the contact email or application link. Always verify postings on official corporate domains.
