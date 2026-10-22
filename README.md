# Explainable Job Posting Risk Analyzer Using NLP and Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)]()
[![Streamlit App](https://img.shields.io/badge/Streamlit-v1.66-FF4B4B.svg)](http://localhost:8502)

An academic machine-learning mini-project developed for coursework evaluation (worth 15 marks). The system analyzes online recruitment advertisements and structured metadata to identify patterns reminiscent of fraudulent job advertisements, delivering probabilistic predictions alongside human-interpretable warning signals, domain heuristics, and explicit research limitations.

---

## 📌 1. Project Overview & Problem Statement

Online recruitment aggregators expose job seekers to employment scams that attempt to extract advance payments, steal banking information, or harvest personal identities. Manual triage across tens of thousands of listings is impractical. 

This project provides an explainable machine learning screening system:
* **Natural Language Processing (NLP)**: TF-IDF feature extraction with sublinear term frequencies, unigrams + bigrams ($n \in [1, 2]$), and stop-word filtering.
* **Structured Tabular Metadata**: Ingestion of company logo indicators, prescreening questions, telecommuting flags, and imputed categorical features.
* **Class Imbalance Mitigation**: Class prevalence is only **4.84%** ($\approx 19.6 : 1$ legitimate to fraud ratio). Models employ cost-sensitive learning (`class_weight='balanced'`) and are evaluated on **PR-AUC (Average Precision)**, **Recall**, and **F1-Score**.
* **Zero Data Leakage Protocol**: Stratified 70/15/15 partitions; all vectorizers and encoders are fitted **strictly on the training split**.
* **Dual-Layer Explainability**: Statistical feature log-odds weights are paired with domain heuristic rules (wire transfers, Telegram/WhatsApp communication, missing identity markers).
* **Interactive Web Applications**: A React risk-scanning interface backed by FastAPI, plus the original 7-tab Streamlit application for EDA, dataset uploads, benchmarks, and error analysis.

---

## 🗂️ 2. Repository Directory Structure

```
Job Posting Risk Analyzer/
├── .gitignore
├── README.md                           # Comprehensive documentation & viva guide
├── requirements.txt                    # Pinned Python package dependencies
├── app.py                              # 7-Tab interactive Streamlit web application
├── train_evaluate.py                   # End-to-end reproducible training & evaluation
├── generate_terminal_report.py         # Formatted CLI summary for presentations
├── generate_figures.py                 # Static publication figure generator
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv       # EMSCAD raw dataset (47.73 MB, 17,880 rows)
│   ├── processed/
│   │   ├── train.csv                   # Stratified training split (12,516 rows, 4.84% fraud)
│   │   ├── val.csv                     # Stratified validation split (2,682 rows, 4.85% fraud)
│   │   └── test.csv                    # Held-out test split (2,682 rows, 4.85% fraud)
│   ├── uploads/
│   │   └── .gitkeep                    # Directory for user-uploaded CSVs (isolated)
│   └── README.md                       # Formal dataset manifest and schema documentation
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, column definitions, thresholds, seeds
│   ├── data_loader.py                  # Ingestion, cleaning, stratified 70/15/15 splitting
│   ├── preprocessing.py                # TF-IDF & OneHot ColumnTransformer pipeline
│   ├── preprocessor.py                 # Preprocessor implementation
│   ├── eda.py                          # Comprehensive EDA & Plotly visualization engine
│   ├── models.py                       # 4 Classifiers + Majority Baseline factories
│   ├── train_evaluate.py               # Package training wrapper
│   ├── evaluation.py                   # Imbalanced metrics, PR curves, ROC curves
│   ├── explainability.py               # Heuristics, risk categorization, token highlighter
│   └── predict.py                      # Production inference & risk scoring engine
├── models/
│   ├── preprocessor.joblib             # Fitted ColumnTransformer (10,033 dimensions)
│   ├── linear_svm.joblib               # Calibrated Linear Support Vector Machine
│   ├── logistic_regression.joblib      # Balanced Logistic Regression
│   ├── multinomial_naive_bayes.joblib  # Multinomial Naive Bayes
│   ├── random_forest.joblib            # Balanced Random Forest
│   └── evaluation_results.json         # Complete benchmark metrics and curve coordinates
├── reports/
│   ├── model_comparison.csv            # Tabular evaluation metrics comparison
│   └── figures/
│       ├── model_comparison_bar.png    # Precision, Recall, F1, PR-AUC comparison
│       ├── confusion_matrices.png      # 2x2 confusion matrices for all models
│       ├── precision_recall_curves.png # PR curves under class imbalance
│       ├── roc_curves.png              # ROC curves
│       └── feature_ablation.png        # Text-only vs Text+Metadata ablation
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                # 8 comprehensive unit & integration tests
└── notebooks/
    └── exploratory_analysis.ipynb      # Clean Jupyter notebook for academic coursework
```

---

## ⚙️ 3. Installation & Setup (Windows PowerShell)

### Step 1: Create and Activate Virtual Environment
```powershell
# Open Windows PowerShell in project root directory
py -3 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (If script execution is blocked on Windows, execute this once:)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Package Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 4. How to Run the Project

### A. Run Automated Unit Tests (Verify System Health)
```powershell
python -m unittest discover -s tests -v
```
*Expected Result:* `Ran 8 tests ... OK` (verifies preprocessor isolation, probability outputs, risk thresholds, and edge cases).

### B. Run Model Training & Evaluation Pipeline
```powershell
python train_evaluate.py
```
*This script will:*
1. Ingest `data/raw/fake_job_postings.csv` (or automatically retrieve it if absent).
2. Clean text, construct metadata flags, and generate leak-free stratified partitions (`data/processed/`).
3. Fit the ColumnTransformer (10,033 dimensions) **strictly on training data**.
4. Train all four classifiers plus the Majority Class baseline with 5-fold stratified cross-validation.
5. Run the feature ablation study (Text-Only vs Text+Metadata).
6. Perform test set error analysis (identifying false positives and false negatives).
7. Save models to `models/` and export `reports/model_comparison.csv`.

### C. Generate Publication Figures
```powershell
python generate_figures.py
```
*Outputs high-resolution PNGs to `reports/figures/` for presentations and coursework submissions.*

### D. Generate Executive Terminal Report
```powershell
python generate_terminal_report.py
```
*Prints the complete statistical summary directly into your console.*

### E. Launch the React Web Application
```powershell
# Install the Python API dependencies (if needed)
pip install -r requirements.txt

# Install and compile the React frontend
cd frontend
npm install
npm run build
cd ..

# Serve the API and compiled frontend together
uvicorn api:app --reload
```
Open **`http://localhost:8000`**. During frontend development, run `npm run dev` from `frontend/`; Vite proxies `/api` requests to the FastAPI server on port 8000.

### F. Launch the Original Streamlit Application
```powershell
streamlit run app.py --server.port 8502
```
Open your browser at: **`http://localhost:8502`**

---

## 📊 5. Benchmark Evaluation Results (Held-Out Test Set)

Evaluated on **2,682 held-out test listings** (130 Fraudulent, 2,552 Legitimate) with frozen preprocessors:

| Classifier Model | Test Prec | Test Recall | Fraud F1 | PR-AUC | Bal Acc | 5-Fold CV F1 | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.00% | 0.00% | 0.00% | 0.0485 | 50.00% | 0.000 ± 0.000 | 0.001s |
| **Multinomial Naive Bayes** | 72.44% | 70.77% | 71.60% | 0.8313 | 84.70% | 0.714 ± 0.007 | 0.022s |
| **Logistic Regression** *(Balanced)* | 57.01% | **93.85%** | 70.93% | 0.9116 | **95.12%** | 0.650 ± 0.009 | 0.897s |
| **Linear SVM** *(Calibrated)* | **93.52%** | 77.69% | **84.87%** | **0.9391** | 88.71% | **0.814 ± 0.006** | 3.872s |
| **Random Forest** *(Balanced)* | 69.44% | 76.92% | 72.99% | 0.8551 | 87.60% | 0.728 ± 0.027 | 1.661s |

### Confusion Matrix Breakdown (Test Set: 130 Scams, 2,552 Legit)
* **Linear SVM**:
  * True Negatives: **2,545** / 2,552 (94.9%)
  * False Positives (False Alarms): **Only 7** / 2,552 (0.3% False Positive Rate)
  * False Negatives (Missed Scams): 29 / 130
  * True Positives (Caught Scams): **101** / 130
* **Logistic Regression**:
  * True Negatives: 2,460 / 2,552
  * False Positives: 92 / 2,552
  * False Negatives: **Only 8** / 130 (Catches nearly 94% of all fraudulent listings!)
  * True Positives: **122** / 130

---

## 🔬 6. Feature Ablation Study: Proving Metadata Value

| Configuration | Fraud Precision | Fraud Recall | Fraud F1 | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Text-Only (TF-IDF alone)** | **68.42%** | 90.00% | **77.74%** | 0.8999 |
| **Combined (Text + Metadata)** | 57.01% | **93.85%** | 70.93% | **0.9116** |
| **Observed Difference ($\Delta$)** | -11.41% | **+3.85%** | -6.81% | **+0.0117** |

*Academic Finding:* Incorporating structured metadata (company logo, screening questions, telecommuting flag) improved PR-AUC by **+0.0117** and caught **+3.85% more scams** (reducing missed scams from 13 to 8). Fraudulent listings disproportionately omit corporate logos and screening questions, providing strong discriminative signals.

---

## 🔍 7. Error Analysis: Why Models Make Mistakes

1. **False Positives (Legitimate ads flagged as fraud):**
   * *Cause:* Commission-based direct sales or telesales roles using high-urgency keywords ("immediate hire", "unlimited earning potential") and small employers who forgot to upload an official company logo.
2. **False Negatives (Fraudulent ads that evaded detection):**
   * *Cause:* Sophisticated scammers who plagiarized legitimate corporate career pages verbatim, altering only an application email address or external contact handle.

---

## 🎓 8. Viva Preparation Guide (Oral Exam Q&A for 15 Marks)

### Q1: Why is raw accuracy a misleading metric for this project?
* **Answer:** The EMSCAD dataset exhibits a **19.6 : 1 class imbalance** (only 4.84% of listings are fraudulent). A trivial baseline predicting every posting as legitimate achieves **95.16% accuracy** while detecting exactly zero scams (Recall = 0.0, F1 = 0.0). We therefore evaluate models using **PR-AUC (Average Precision)**, **Recall**, and **F1-Score**.

### Q2: How did you strictly prevent data leakage?
* **Answer:** We partitioned the raw data into Stratified Train (70%), Validation (15%), and Test (15%) splits *before* any feature extraction took place. All `TfidfVectorizer` vocabulary indexing, sublinear scaling, and `OneHotEncoder` categories were fitted **strictly on the 12,516 training samples**. Validation and Test sets were only transformed using the pre-fitted objects. Furthermore, unique identifiers (`job_id`) and target labels were excluded from the feature space.

### Q3: Why did you choose Linear SVM and Logistic Regression?
* **Answer:** With 10,000 TF-IDF features, text classification operates in a very high-dimensional, sparse feature space. Linear classifiers form convex decision boundaries with strong regularization ($L_2$ penalty), train in seconds, and generalize well without overfitting. In addition, Logistic Regression coefficients provide direct log-odds interpretability.

### Q4: How did you extract probabilities from Linear SVM?
* **Answer:** Standard Linear SVM outputs signed distance margins, not probabilities. We applied Platt scaling using scikit-learn's `CalibratedClassifierCV(method='sigmoid', cv=3)`, which fits a calibrated logistic regression model over the SVM decision values to output valid posterior probabilities $[0.0, 1.0]$.

### Q5: What are the primary cybersecurity limitations of this system?
* **Answer:** The EMSCAD dataset is historical and captures traditional scams (check cashing, wire transfers, data entry). Modern threat actors leverage Generative AI (LLMs) to generate pristine corporate descriptions, which can evade keyword-based NLP models. Therefore, this tool is designed as a **preliminary screening aid**, not definitive legal verification.

---

## 📜 9. License & Academic Integrity
* **Code License:** MIT License
* **Dataset License:** CC BY-NC-SA 4.0 (University of the Aegean)
* This project is developed for educational and research evaluation. It does not provide legal advice or corporate guarantee.
