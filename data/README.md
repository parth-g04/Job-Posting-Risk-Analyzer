# Dataset Manifest: Employment Scam Aegean Dataset (EMSCAD)

## 1. Overview & Provenance
* **Dataset Name**: Employment Scam Aegean Dataset (EMSCAD) / Real or Fake Job Posting Prediction
* **Original Creators**: University of the Aegean, Laboratory of Information & Communication Systems Security (ICCS)
* **Kaggle Distributions**:
  * [Amruth Jithraj / Recruitment Scam](https://www.kaggle.com/datasets/amruthjithrajvr/recruitment-scam)
  * [Shivam Bansal / Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
* **Primary License**: CC BY-NC-SA 4.0 / Open Database License (Academic Research and Educational Use)
* **Local Storage Location**:
  * Raw archive: `data/raw/fake_job_postings.csv` (File size: 47.73 MB)
  * Processed partitions: `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`
* **Retrieval Timestamp**: October 2026 (Verified MD5 and schema validation)

---

## 2. Dataset Schema & Column Specification

The dataset contains **17,880 observations** and **18 features**:

| Column Name | Data Type | Expected Content | Missing Count | Handling Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `job_id` | Integer | Unique identifier per advertisement | 0 | **Excluded** from model features to prevent identifier memorization / target leakage. |
| `title` | String | Job position headline | 0 | Text component concatenated into `full_text`. |
| `location` | String | Geographical location (Country, State, City) | 346 | Handled in EDA; extracted country indicator where useful. |
| `department` | String | Organizational department | 11,547 | Highly sparse; optional metadata. |
| `salary_range` | String | Advertised compensation range (e.g. 50k-80k) | 15,012 | Converted to binary indicator `salary_specified` (0/1). |
| `company_profile`| String | Company background, mission, founding | 3,308 | Key discriminator! Extracted to `has_company_profile` (0/1) & text component. |
| `description` | String | Core responsibilities and role summary | 1 | Core text component concatenated into `full_text`. |
| `requirements` | String | Qualifications, technical skills, degrees | 2,695 | Text component concatenated into `full_text`. |
| `benefits` | String | Health insurance, 401(k), bonus, PTO | 7,210 | Text component concatenated into `full_text`. |
| `telecommuting` | Binary (0/1) | Work-from-home flag | 0 | Passed as structured binary feature. |
| `has_company_logo`| Binary (0/1)| Corporate logo attached flag | 0 | Crucial metadata discriminator! (Scams rarely have logos). |
| `has_questions` | Binary (0/1) | Prescreening questions flag | 0 | Passed as structured binary feature. |
| `employment_type`| Categorical | Full-time, Part-time, Contract, Temporary | 3,471 | Imputed with `"Unknown"`, One-Hot Encoded. |
| `required_experience`| Categorical| Entry level, Mid-Senior, Executive, etc. | 7,050 | Imputed with `"Unknown"`, One-Hot Encoded. |
| `required_education` | Categorical| Bachelor's, High School, Master's, etc. | 7,605 | Imputed with `"Unknown"`, One-Hot Encoded. |
| `industry` | Categorical | Business sector (IT, Healthcare, etc.) | 4,903 | Used in EDA and sectoral fraud distribution. |
| `function` | Categorical | Job operational function (Sales, Engineering)| 6,455 | Used in EDA. |
| `fraudulent` | Binary (0/1) | **Target Ground Truth**: 0 = Legitimate, 1 = Fraud | 0 | Target variable used for supervised classification. |

---

## 3. Class Imbalance Characteristics

* **Legitimate Postings (Class 0)**: 17,014 listings (**95.16%**)
* **Fraudulent Postings (Class 1)**: 866 listings (**4.84%**)
* **Imbalance Ratio**: $\approx 19.6 : 1$

### Academic Implication for Model Evaluation:
* **Accuracy is a deceptive metric**: A trivial dummy classifier that unconditionally predicts class 0 achieves **95.16% accuracy** while capturing **0% of scams** (Recall = 0.0, F1 = 0.0).
* Therefore, the evaluation protocol strictly centers on **PR-AUC (Average Precision)**, **Fraud Precision**, **Fraud Recall**, and **Fraud F1-Score**.

---

## 4. Duplicate & Near-Duplicate Analysis
* **Exact Row Duplicates**: 0 (all 17,880 rows possess distinct identifiers).
* **Duplicate Job Descriptions**: 3,078 occurrences ($\approx 17.2\%$).
  * *Etiology*: Staffing firms recruiting across multiple cities and scam operators re-using boilerplate scam copy.
  * *Methodological Consideration*: A standard random split could place near-identical descriptions in both train and test partitions. Our project evaluates both conventional Stratified Splitting and investigates Group-aware / Deduplicated considerations to document generalization impact.

---

## 5. Manual Download Instructions (Fallback)

If automated downloading is interrupted or network restrictions apply:
1. Visit [Kaggle EMSCAD Page](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
2. Download `archive.zip` and extract `fake_job_postings.csv`.
3. Place the file at: `data/raw/fake_job_postings.csv`.
4. Validate using: `python -m src.data_loader`.
