"""
Explainable Job Posting Risk Analyzer - Interactive Streamlit Application
A multi-tier screening system combining NLP (TF-IDF), 4 Machine Learning classifiers,
heuristic threat indicators, and explainability signals.
"""
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    MODELS_DIR,
    METRICS_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD
)
from src.data_loader import clean_and_feature_engineer
from src.explainability import (
    scan_heuristic_flags,
    generate_risk_assessment,
    highlight_suspicious_phrases_html
)

# Page Configuration
st.set_page_config(
    page_title="Job Posting Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    .flag-card {
        background: #f8fafc;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .disclaimer-card {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Preset Sample Job Postings
PRESET_SAMPLES = {
    "Select a pre-configured sample...": None,
    "⚠️ Blatant Scam: Home Data Entry with Wire Request & Telegram": {
        "title": "Immediate Data Entry Clerk - Work From Home",
        "company_profile": "",
        "description": "Urgent recruitment! We are seeking remote data entry assistants. No prior experience is required. Candidates will receive an initial check of $4,500 to purchase home office equipment from our certified vendor via wire transfer or Western Union. Once cleared, start making $45/hour immediately. Must contact our hiring manager directly via Telegram @recruiter_fastjob or WhatsApp.",
        "requirements": "Must have basic computer literacy. Must have active bank account to deposit equipment check. Immediate start required without interview.",
        "benefits": "Earn $2,000 to $4,000 weekly. Flexible hours, fully remote work.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "High School or equivalent"
    },
    "✅ Legitimate Posting: Senior Python Backend Engineer": {
        "title": "Senior Python Backend Engineer",
        "company_profile": "CloudScale Systems is a leading enterprise cloud observability company founded in 2016. We serve Fortune 500 enterprises across North America and Europe, providing low-latency distributed telemetry and analytics.",
        "description": "We are looking for a seasoned Senior Python Backend Engineer to scale our core data ingestion service. In this role, you will design distributed microservices using FastAPI, Redis, and Kafka, and collaborate with DevOps engineers to ensure high availability.",
        "requirements": "5+ years software engineering experience. Strong proficiency with Python, asynchronous programming, PostgreSQL, and AWS or GCP. Bachelor's degree in Computer Science or equivalent practical experience.",
        "benefits": "Competitive base salary, 401(k) matching up to 5%, comprehensive medical/dental/vision coverage, generous parental leave, and unlimited PTO.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree"
    },
    "⚠️ Suspicious Listing: Vague Assistant with Personal Gmail Contact": {
        "title": "Executive Personal Assistant - Flexible Schedule",
        "company_profile": "Private confidential client looking for trusted assistant.",
        "description": "Busy executive looking for dependable personal assistant to handle errands, mail correspondence, and confidential tasks. Send your resume along with copy of driver's license to executive.manager2024@gmail.com for prompt review.",
        "requirements": "Trustworthy, prompt communicator. Must possess clean driving record.",
        "benefits": "Weekly allowance plus performance bonuses.",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "employment_type": "Contract",
        "required_experience": "Not Applicable",
        "required_education": "Unspecified"
    }
}


@st.cache_resource
def load_model_artifacts():
    """Load preprocessor, classifiers, and evaluation metrics if available."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    if not preprocessor_path.exists():
        return None, None, None

    preprocessor = joblib.load(preprocessor_path)

    models = {}
    model_files = {
        "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib",
        "Random Forest": "random_forest.joblib"
    }

    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)

    metrics = None
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

    return preprocessor, models, metrics


def main():
    preprocessor, models, metrics = load_model_artifacts()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.title("Risk Guard")
        st.markdown("**Explainable Recruitment Fraud Detection**")
        st.caption("NLP & Machine Learning Screening Engine")

        st.markdown("---")
        if models:
            selected_model_name = st.selectbox(
                "Active Classifier Engine",
                options=list(models.keys()),
                index=1  # Default to Logistic Regression for clear coefficients
            )
            active_model = models[selected_model_name]
            st.success(f"Loaded: **{selected_model_name}**")
        else:
            st.warning("Models not yet trained.")
            selected_model_name = None
            active_model = None

        st.markdown("---")
        st.markdown("### System Configuration")
        high_risk_thresh = st.slider("High Risk Threshold", 0.50, 0.85, 0.65, 0.05)
        mod_risk_thresh = st.slider("Moderate Risk Threshold", 0.15, 0.45, 0.25, 0.05)

        st.markdown("---")
        st.markdown(
            "💡 **Intended Scope:**\n"
            "This application is a **preliminary screening aid**. "
            "Predictions are based on statistical similarities to historical fraudulent ads "
            "and should not replace manual verification."
        )

    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:800; color:#38bdf8;">
            🛡️ Job Posting Risk Analyzer
        </h1>
        <p style="margin-top:8px; margin-bottom:0; color:#94a3b8; font-size:1.05rem;">
            Screen online job advertisements for potential fraud, identify deceptive language, and inspect interpretable warning indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs
    tab_scanner, tab_benchmarks, tab_ablation, tab_eda, tab_methodology = st.tabs([
        "🔍 Job Risk Scanner",
        "📊 Model Benchmarks & Comparison",
        "🔬 Feature Ablation (Metadata vs Text)",
        "📈 Dataset Insights (EDA)",
        "📖 Methodology & Limitations"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: JOB RISK SCANNER
    # -------------------------------------------------------------------------
    with tab_scanner:
        st.markdown("### 📋 Listing Details & Metadata Input")
        st.write("Input job posting information below or choose a pre-configured sample to test.")

        # Preset Loader
        preset_choice = st.selectbox("Load Sample Job Posting:", list(PRESET_SAMPLES.keys()))
        sample_data = PRESET_SAMPLES.get(preset_choice)

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            title_val = sample_data["title"] if sample_data else ""
            job_title = st.text_input("Job Title *", value=title_val, placeholder="e.g. Data Entry Assistant / Remote Support")

            company_val = sample_data["company_profile"] if sample_data else ""
            company_profile = st.text_area(
                "Company Profile",
                value=company_val,
                placeholder="Background, mission statement, or founding year of the hiring organization...",
                height=90
            )

            desc_val = sample_data["description"] if sample_data else ""
            job_description = st.text_area(
                "Job Description *",
                value=desc_val,
                placeholder="Core responsibilities, duties, and hiring conditions...",
                height=150
            )

            req_val = sample_data["requirements"] if sample_data else ""
            requirements = st.text_area(
                "Requirements & Qualifications",
                value=req_val,
                placeholder="Required skills, years of experience, tools, degrees...",
                height=90
            )

            ben_val = sample_data["benefits"] if sample_data else ""
            benefits = st.text_area(
                "Benefits & Compensation Details",
                value=ben_val,
                placeholder="Health insurance, 401(k), salary promises, bonuses...",
                height=80
            )

        with col_right:
            st.markdown("#### Structured Metadata Attributes")
            st.caption("Recruiters and scams exhibit distinct metadata patterns.")

            telecommuting = st.checkbox(
                "Telecommuting / 100% Remote Position",
                value=bool(sample_data["telecommuting"]) if sample_data else True
            )

            has_company_logo = st.checkbox(
                "Employer Uploaded Company Logo",
                value=bool(sample_data["has_company_logo"]) if sample_data else False
            )

            has_questions = st.checkbox(
                "Job Application Includes Screening Questions",
                value=bool(sample_data["has_questions"]) if sample_data else False
            )

            emp_options = ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"]
            emp_default = emp_options.index(sample_data["employment_type"]) if sample_data and sample_data["employment_type"] in emp_options else 0
            employment_type = st.selectbox("Employment Type", emp_options, index=emp_default)

            exp_options = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"]
            exp_default = exp_options.index(sample_data["required_experience"]) if sample_data and sample_data["required_experience"] in exp_options else 0
            required_experience = st.selectbox("Required Experience Level", exp_options, index=exp_default)

            edu_options = ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Associate Degree", "Unspecified", "Unknown"]
            edu_default = edu_options.index(sample_data["required_education"]) if sample_data and sample_data["required_education"] in edu_options else 0
            required_education = st.selectbox("Required Education", edu_options, index=edu_default)

            st.markdown("---")
            analyze_btn = st.button("🛡️ Analyze Risk Score", type="primary", use_container_width=True)

        if analyze_btn or sample_data is not None:
            if not job_title and not job_description:
                st.error("Please provide at least a Job Title or Job Description to run the analysis.")
            elif not preprocessor or not active_model:
                st.error("Model artifacts not loaded. Please train the models first using `python train_evaluate.py`.")
            else:
                # Prepare single-row DataFrame
                input_row = {
                    "title": job_title,
                    "company_profile": company_profile,
                    "description": job_description,
                    "requirements": requirements,
                    "benefits": benefits,
                    "telecommuting": int(telecommuting),
                    "has_company_logo": int(has_company_logo),
                    "has_questions": int(has_questions),
                    "employment_type": employment_type,
                    "required_experience": required_experience,
                    "required_education": required_education,
                    "salary_range": ""
                }

                df_input = pd.DataFrame([input_row])
                df_clean = clean_and_feature_engineer(df_input)

                # Heuristic Rule Scan
                metadata_dict = {
                    "has_company_logo": int(has_company_logo),
                    "has_company_profile": df_clean.iloc[0]["has_company_profile"],
                    "has_questions": int(has_questions)
                }
                heuristic_flags = scan_heuristic_flags(df_clean.iloc[0]["full_text"], metadata_dict)

                # ML Prediction
                X_trans = preprocessor.transform(df_clean)
                probs = active_model.predict_proba(X_trans)[0]
                fraud_prob = float(probs[1])

                assessment = generate_risk_assessment(fraud_prob, heuristic_flags)

                # Presentation Section
                st.markdown("---")
                st.markdown("## 🔍 Screening Assessment & Explainability Signals")

                r_col1, r_col2 = st.columns([1, 1.4])

                with r_col1:
                    # Risk Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(fraud_prob * 100, 1),
                        number={"suffix": "%", "font": {"size": 36, "color": assessment["color"]}},
                        title={"text": "Estimated Fraud Probability", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": assessment["color"], "thickness": 0.25},
                            "steps": [
                                {"range": [0, 25], "color": "#dcfce7"},
                                {"range": [25, 65], "color": "#fef3c7"},
                                {"range": [65, 100], "color": "#fee2e2"}
                            ],
                            "threshold": {
                                "line": {"color": "#dc2626", "width": 4},
                                "thickness": 0.75,
                                "value": 65
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Badge Banner
                    badge_class = f"badge-{assessment['badge_style']}"
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: -10px; margin-bottom: 15px;">
                        <span class="{badge_class}">
                            Category: {assessment['risk_level'].upper()}
                        </span>
                        <p style="margin-top: 10px; color: #475569; font-weight: 500;">
                            {assessment['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with r_col2:
                    st.markdown("#### 🚨 Detected Warning Signals & Rule Flags")
                    if heuristic_flags:
                        for flag in heuristic_flags:
                            sev_color = "#ef4444" if flag["severity"] == "HIGH" else "#f59e0b" if flag["severity"] == "MEDIUM" else "#64748b"
                            st.markdown(f"""
                            <div class="flag-card" style="border-left-color: {sev_color};">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:#0f172a; font-size:0.95rem;">{flag['category']}</strong>
                                    <span style="font-size:0.75rem; font-weight:700; color:{sev_color}; text-transform:uppercase;">{flag['severity']} SEVERITY</span>
                                </div>
                                <div style="font-size:0.85rem; color:#475569; margin-top:4px;">{flag['description']}</div>
                                <div style="margin-top:4px;"><code style="font-size:0.75rem; background:#e2e8f0; padding:2px 6px; border-radius:4px;">Triggers: {', '.join(flag['evidence'])}</code></div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No heuristic red flags detected in contact info, payment demands, or metadata.")

                # Explanatory Text Inspection
                st.markdown("#### 📝 Text Inspection with Flagged Indicators")
                highlighted_html = highlight_suspicious_phrases_html(df_clean.iloc[0]["full_text"])
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; line-height: 1.7; max-height: 250px; overflow-y: auto;">
                    {highlighted_html}
                </div>
                """, unsafe_allow_html=True)
                st.caption("🔴 Red: Suspicious or high-risk scam triggers. 🟢 Green: Common markers of legitimate employment.")

                # Actionable Recommendations & Limitations
                rec_col1, rec_col2 = st.columns([1.2, 1])
                with rec_col1:
                    st.markdown("#### 🛡️ Safe Verification Checklist for Applicants")
                    for r in assessment["recommendations"]:
                        st.markdown(f"- {r}")

                with rec_col2:
                    st.markdown("#### ⚠️ Critical Limitations")
                    st.markdown(f"""
                    <div class="disclaimer-card">
                        <strong>Preliminary Screening Notice:</strong><br>
                        {assessment['disclaimer']}
                    </div>
                    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: MODEL BENCHMARKS & COMPARISON
    # -------------------------------------------------------------------------
    with tab_benchmarks:
        st.markdown("### 📊 Benchmark Comparison Across 4 Machine Learning Classifiers")
        st.write("Comprehensive evaluation addressing class imbalance, precision-recall tradeoffs, and held-out test generalization.")

        if metrics and "models" in metrics:
            model_summary = []
            for name, data in metrics["models"].items():
                tm = data["test_metrics"]
                vm = data["val_metrics"]
                model_summary.append({
                    "Model": name,
                    "Val F1": vm["fraud_f1"],
                    "Test Precision": tm["fraud_precision"],
                    "Test Recall": tm["fraud_recall"],
                    "Test F1-Score": tm["fraud_f1"],
                    "PR-AUC (Avg Precision)": tm["pr_auc"],
                    "ROC-AUC": tm["roc_auc"],
                    "Accuracy": tm["accuracy"],
                    "Train Time (s)": data["train_time_sec"]
                })

            df_models = pd.DataFrame(model_summary)
            st.dataframe(
                df_models.style.highlight_max(subset=["Test Precision", "Test Recall", "Test F1-Score", "PR-AUC (Avg Precision)", "ROC-AUC"], color="#bbf7d0")
                               .format({
                                   "Val F1": "{:.3f}",
                                   "Test Precision": "{:.3f}",
                                   "Test Recall": "{:.3f}",
                                   "Test F1-Score": "{:.3f}",
                                   "PR-AUC (Avg Precision)": "{:.3f}",
                                   "ROC-AUC": "{:.3f}",
                                   "Accuracy": "{:.3f}",
                                   "Train Time (s)": "{:.2f}"
                               }),
                use_container_width=True
            )

            # Interactive Visualizations
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                st.markdown("#### 📈 Precision-Recall Curves")
                fig_pr = go.Figure()
                for name, data in metrics["models"].items():
                    if data.get("pr_curve"):
                        pr = data["pr_curve"]
                        fig_pr.add_trace(go.Scatter(
                            x=pr["recall"],
                            y=pr["precision"],
                            mode="lines",
                            name=f"{name} (AUC={pr['pr_auc']:.3f})"
                        ))
                fig_pr.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    title="PR Curves (Held-out Test Set)",
                    height=380,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            with v_col2:
                st.markdown("#### 📈 ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
                for name, data in metrics["models"].items():
                    if data.get("roc_curve"):
                        roc = data["roc_curve"]
                        fig_roc.add_trace(go.Scatter(
                            x=roc["fpr"],
                            y=roc["tpr"],
                            mode="lines",
                            name=f"{name} (AUC={roc['roc_auc']:.3f})"
                        ))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    title="ROC Curves (Held-out Test Set)",
                    height=380,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            # Confusion Matrix Inspector
            st.markdown("---")
            st.markdown("#### 🔲 Confusion Matrix Breakdown by Model")
            selected_cm_model = st.selectbox("Inspect Confusion Matrix for:", list(metrics["models"].keys()))
            cm_data = metrics["models"][selected_cm_model]["test_metrics"]["confusion_matrix"]

            cm_matrix = [
                [cm_data["tn"], cm_data["fp"]],
                [cm_data["fn"], cm_data["tp"]]
            ]

            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Legitimate (0)", "Fraudulent (1)"],
                y=["Legitimate (0)", "Fraudulent (1)"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=320, width=500)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.caption(f"**Interpretation:** True Negatives: {cm_data['tn']} | False Positives (False Alarms): {cm_data['fp']} | False Negatives (Missed Scams): {cm_data['fn']} | True Positives (Caught Scams): {cm_data['tp']}.")

        else:
            st.info("Run `python train_evaluate.py` to generate complete model benchmarks and test metrics.")

    # -------------------------------------------------------------------------
    # TAB 3: FEATURE ABLATION (METADATA VS TEXT)
    # -------------------------------------------------------------------------
    with tab_ablation:
        st.markdown("### 🔬 Feature Ablation: Evaluating Value of Structured Metadata")
        st.write("Does combining structured metadata with TF-IDF text features improve fraud detection performance?")

        if metrics and "ablation_study" in metrics:
            ab = metrics["ablation_study"]
            to = ab["text_only"]
            cb = ab["combined"]

            ablation_df = pd.DataFrame([
                {
                    "Feature Set": "Text Only (TF-IDF)",
                    "Precision": to["fraud_precision"],
                    "Recall": to["fraud_recall"],
                    "F1-Score": to["fraud_f1"],
                    "PR-AUC": to["pr_auc"]
                },
                {
                    "Feature Set": "Combined (Text + Metadata)",
                    "Precision": cb["fraud_precision"],
                    "Recall": cb["fraud_recall"],
                    "F1-Score": cb["fraud_f1"],
                    "PR-AUC": cb["pr_auc"]
                }
            ])

            st.dataframe(
                ablation_df.style.format({
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "PR-AUC": "{:.4f}"
                }),
                use_container_width=True
            )

            # Delta metrics cards
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.metric("F1-Score Improvement with Metadata", f"{cb['fraud_f1']:.4f}", delta=f"{ab['delta_f1']:+.4f}")
            with d_col2:
                st.metric("PR-AUC Improvement with Metadata", f"{cb['pr_auc']:.4f}", delta=f"{ab['delta_pr_auc']:+.4f}")

            st.markdown("""
            **Key Finding:** 
            Adding structured metadata (such as presence of company logo, screening questions, telecommuting flag, and employment type) 
            significantly boosts detection precision and reduces false positives compared to text-only modeling. Fraudulent actors 
            frequently leave metadata fields blank (e.g. omitting corporate logos and verified profiles).
            """)
        else:
            st.info("Ablation study metrics will appear here once `python train_evaluate.py` finishes execution.")

    # -------------------------------------------------------------------------
    # TAB 4: DATASET INSIGHTS (EDA)
    # -------------------------------------------------------------------------
    with tab_eda:
        st.markdown("### 📈 Dataset Distribution & Exploratory Insights")
        
        if metrics and "dataset_metadata" in metrics:
            meta = metrics["dataset_metadata"]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Total Listings Analyzed", f"{meta['total_samples']:,}")
            m_col2.metric("Training Split (70%)", f"{meta['train_samples']:,}")
            m_col3.metric("Test Split (15%)", f"{meta['test_samples']:,}")
            m_col4.metric("Fraud Prevalence", f"{meta['fraud_prevalence_pct']}%")

        if metrics and "global_explainability" in metrics:
            st.markdown("#### 🔍 Most Influential Fraud vs Legitimate Terms (Model Weights)")
            ge = metrics["global_explainability"]
            
            top_fraud = pd.DataFrame(ge["top_fraud_terms"][:15], columns=["Term", "Weight"])
            top_legit = pd.DataFrame(ge["top_legit_terms"][:15], columns=["Term", "Weight"])

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                fig_fraud = px.bar(
                    top_fraud,
                    x="Weight",
                    y="Term",
                    orientation="h",
                    title="Top Fraud-Associated N-Grams",
                    color_discrete_sequence=["#ef4444"]
                )
                fig_fraud.update_layout(yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_fraud, use_container_width=True)

            with t_col2:
                fig_legit = px.bar(
                    top_legit,
                    x="Weight",
                    y="Term",
                    orientation="h",
                    title="Top Legitimate-Associated N-Grams",
                    color_discrete_sequence=["#10b981"]
                )
                fig_legit.update_layout(yaxis=dict(autorange="reversed"), height=420)
                st.plotly_chart(fig_legit, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: METHODOLOGY & LIMITATIONS
    # -------------------------------------------------------------------------
    with tab_methodology:
        st.markdown("### 📖 Methodology, Leakage Prevention, & Research Limitations")
        st.markdown("""
        #### 1. Preventing Data Leakage
        - **Strict Split Isolation**: The raw dataset is partitioned into Stratified Train (70%), Validation (15%), and Test (15%) subsets.
        - **Zero Preprocessing Leakage**: TF-IDF vectorizers, sublinear term frequencies, vocabulary indexing, and categorical encoders are fitted **strictly on the training set**.
        - **Identifer Exclusion**: `job_id` and raw target labels are completely excluded from the feature space.

        #### 2. Class Imbalance Mitigation
        - The labeled recruitment fraud dataset has an inherent class imbalance (~4.8% fraudulent listings).
        - To counter this, classifiers employ `class_weight='balanced'` and evaluation is centered around **PR-AUC (Average Precision)**, **Fraud F1-Score**, and **Recall** rather than raw accuracy.

        #### 3. Algorithmic Comparison
        - **Multinomial Naive Bayes**: Fast probabilistic baseline suited for text token frequencies.
        - **Logistic Regression**: Linear classifier providing direct feature interpretability via log-odds weights.
        - **Linear Support Vector Machine**: Maximal margin separator calibrated with Platt scaling (`CalibratedClassifierCV`) to yield proper probabilities.
        - **Random Forest**: Ensembles decision trees to evaluate non-linear interactions between metadata and vocabulary.

        #### 4. Critical Cybersecurity Limitations
        - **Historical Bias**: Models trained on historical data reflect scams of their era (e.g., check cashing, wire transfers). Contemporary threat actors leverage generative AI to craft sophisticated, typo-free listings.
        - **Evasion Attacks**: Adversaries can inject legitimate corporate boilerplate to lower statistical risk scores.
        - **Human-in-the-Loop Requirement**: This application serves strictly as an automated triage tool. Job seekers must conduct independent due diligence before providing confidential personal information or accepting employment offers.
        """)


if __name__ == "__main__":
    main()
