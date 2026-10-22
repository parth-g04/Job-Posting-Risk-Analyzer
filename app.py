"""
Explainable Job Posting Risk Analyzer — Academic & Industry Screening Application
7-Tab Unified Web Interface:
1. Overview
2. Job Risk Scanner
3. Dataset Explorer & EDA
4. Upload Your Dataset
5. Model Benchmarks
6. Explainability & Error Analysis
7. Methodology & Limitations
"""
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    METRICS_FILE,
    RAW_DATA_FILE,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD
)
from src.predict import get_prediction_engine
from src.eda import (
    get_dataset_overview_stats,
    plot_target_distribution,
    plot_missing_values,
    plot_metadata_correlation,
    plot_categorical_feature_by_target,
    plot_text_length_distributions,
    compute_top_ngrams_by_class
)

# Page configuration
st.set_page_config(
    page_title="Job Posting Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Academic Polish
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


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
        "company_profile": "CloudScale Systems is an enterprise cloud observability company founded in 2016. We serve global technology enterprises, providing low-latency distributed telemetry and analytics across hybrid cloud infrastructures.",
        "description": "We are looking for a Senior Python Backend Engineer to scale our core data ingestion service. In this role, you will design distributed microservices using FastAPI, Redis, and Kafka, and collaborate with DevOps engineers to ensure high availability.",
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


@st.cache_data
def load_raw_dataset_cached():
    """Load default raw EMSCAD dataset for EDA."""
    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE), RAW_DATA_FILE.stat().st_size
    return None, 0


@st.cache_data
def load_metrics_and_comparison():
    """Load evaluation JSON and model comparison CSV."""
    metrics = None
    comparison_df = None

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

    comp_csv = REPORTS_DIR / "model_comparison.csv"
    if comp_csv.exists():
        comparison_df = pd.read_csv(comp_csv)

    return metrics, comparison_df


def main():
    engine = get_prediction_engine()
    raw_df, file_size = load_raw_dataset_cached()
    metrics, comp_df = load_metrics_and_comparison()

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.title("Risk Guard")
        st.caption("Explainable Recruitment Fraud Screening")
        st.markdown("---")

        available_models = engine.get_available_models() if engine.is_ready else ["Linear SVM"]
        default_idx = available_models.index("Linear SVM") if "Linear SVM" in available_models else 0
        selected_model = st.selectbox("Active Screening Classifier", available_models, index=default_idx)

        st.markdown("#### Configurable Risk Thresholds")
        high_thresh = st.slider("High Risk Threshold (Alert)", 0.50, 0.85, 0.65, 0.05,
                                help="Predictions at or above this calibrated probability trigger a High Risk rating.")
        low_thresh = st.slider("Moderate Risk Threshold (Warning)", 0.15, 0.45, 0.25, 0.05,
                               help="Predictions above this threshold enter the Moderate / Warning category.")

        st.markdown("---")
        st.info("💡 **Academic Mini-Project (15 Marks)**\n\nEMSCAD Dataset (17,880 listings)\nClass Imbalance: 4.84% Fraud\nLeak-Free Stratified 70/15/15 Split")

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem; font-weight:800; color:#38bdf8;">
            🛡️ Explainable Job Posting Risk Analyzer
        </h1>
        <p style="margin-top:8px; margin-bottom:0; color:#cbd5e1; font-size:1.05rem;">
            A multi-tier machine learning screening tool combining NLP, calibrated probability models,
            heuristic threat indicators, and explainability signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Seven Clearly Defined Tabs
    tabs = st.tabs([
        "📌 Overview",
        "🔍 Job Risk Scanner",
        "📊 Dataset Explorer & EDA",
        "📂 Upload Your Dataset",
        "📈 Model Benchmarks",
        "🔬 Explainability & Error Analysis",
        "📖 Methodology & Limitations"
    ])

    tab_overview, tab_scanner, tab_eda, tab_upload, tab_benchmarks, tab_explain, tab_methodology = tabs

    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab_overview:
        st.markdown("### 🎯 Problem Statement & Threat Model")
        st.markdown("""
        Online recruitment aggregators expose job seekers to sophisticated employment scams designed to 
        extract money (advance fee/equipment purchase scams), siphon banking credentials, or steal identities. 
        Because manual verification of thousands of daily advertisements is impractical, this project delivers 
        an automated, explainable screening system.
        """)

        o_col1, o_col2, o_col3, o_col4 = st.columns(4)
        o_col1.metric("Total EMSCAD Postings", "17,880")
        o_col2.metric("Legitimate Listings", "17,014 (95.16%)")
        o_col3.metric("Fraudulent Listings", "866 (4.84%)")
        o_col4.metric("Class Imbalance Ratio", "19.6 : 1")

        st.markdown("---")
        st.markdown("### 🏗️ End-to-End System Architecture")
        st.markdown("""
        ```mermaid
        graph LR
            A[Job Advertisement Input] --> B[Text Cleaner & Normalizer]
            A --> C[Metadata Extractor]
            B --> D[TF-IDF N-Gram Vectorizer]
            C --> E[One-Hot & Binary Transformer]
            D & E --> F[ColumnTransformer Sparse Matrix 10,033 Dims]
            F --> G[Calibrated Classifier Linear SVM / LogReg]
            A --> H[Domain Heuristic Rules Wire/Crypto/Telegram]
            G & H --> I[Composite Risk Assessment Engine]
            I --> J[Streamlit Explainable Output & Highlighting]
        ```
        """)

        st.markdown("### 🏆 Core Methodological Contributions")
        st.markdown("""
        1. **Zero Data Leakage Protocol**: TF-IDF vocabulary, n-gram bounds, and categorical encodings are fitted **strictly on the 70% training split**.
        2. **Imbalance-Aware Classifiers**: Uses `class_weight='balanced'` and evaluates primarily on **PR-AUC (Average Precision)**, **Fraud Recall**, and **Fraud F1-Score**.
        3. **Dual-Layer Explainability**: Combines statistical model weights (log-odds impact) with concrete cybersecurity heuristic rules.
        4. **Separation of Concerns**: Screening probability is decoupled from heuristic triggers, preventing false confidence.
        """)

    # =========================================================================
    # TAB 2: JOB RISK SCANNER
    # =========================================================================
    with tab_scanner:
        st.markdown("### 🔍 Live Job Posting Risk Scanner")
        st.write("Input listing text and metadata or load a realistic pre-configured scenario.")

        preset_name = st.selectbox("Load Sample Job Posting:", list(PRESET_SAMPLES.keys()))
        preset_data = PRESET_SAMPLES.get(preset_name)

        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            job_title = st.text_input("Job Title *", value=preset_data["title"] if preset_data else "", placeholder="e.g. Remote Data Entry / Senior Backend Engineer")
            company_profile = st.text_area("Company Profile", value=preset_data["company_profile"] if preset_data else "", placeholder="Corporate background, founding date, mission...", height=90)
            job_description = st.text_area("Job Description *", value=preset_data["description"] if preset_data else "", placeholder="Core duties, responsibilities, tasks...", height=150)
            requirements = st.text_area("Requirements & Qualifications", value=preset_data["requirements"] if preset_data else "", placeholder="Required experience, tools, education...", height=90)
            benefits = st.text_area("Benefits & Compensation", value=preset_data["benefits"] if preset_data else "", placeholder="Salary, healthcare, 401(k), bonuses...", height=80)

        with col_right:
            st.markdown("#### Structured Metadata")
            telecommuting = st.checkbox("Telecommuting / 100% Remote", value=bool(preset_data["telecommuting"]) if preset_data else True)
            has_company_logo = st.checkbox("Employer Attached Company Logo", value=bool(preset_data["has_company_logo"]) if preset_data else False)
            has_questions = st.checkbox("Application Has Screening Questions", value=bool(preset_data["has_questions"]) if preset_data else False)

            emp_options = ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"]
            emp_idx = emp_options.index(preset_data["employment_type"]) if preset_data and preset_data["employment_type"] in emp_options else 0
            employment_type = st.selectbox("Employment Type", emp_options, index=emp_idx)

            exp_options = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"]
            exp_idx = exp_options.index(preset_data["required_experience"]) if preset_data and preset_data["required_experience"] in exp_options else 0
            required_experience = st.selectbox("Required Experience Level", exp_options, index=exp_idx)

            edu_options = ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Associate Degree", "Unspecified", "Unknown"]
            edu_idx = edu_options.index(preset_data["required_education"]) if preset_data and preset_data["required_education"] in edu_options else 0
            required_education = st.selectbox("Required Education", edu_options, index=edu_idx)

            st.markdown("---")
            scan_btn = st.button("🛡️ Execute Risk Screening", type="primary", use_container_width=True)

        if scan_btn or preset_data is not None:
            if not job_title and not job_description:
                st.error("Please provide at least a Job Title or Job Description.")
            elif not engine.is_ready:
                st.error("Model artifacts not loaded. Please execute `python train_evaluate.py`.")
            else:
                input_data = {
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

                assessment = engine.predict(
                    input_data,
                    model_name=selected_model,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh
                )

                st.markdown("---")
                st.markdown("## 🔍 Screening Assessment & Explainability Signals")

                r1, r2 = st.columns([1, 1.4])
                with r1:
                    prob = assessment["ml_probability"]
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob * 100, 1),
                        number={"suffix": "%", "font": {"size": 36, "color": assessment["color"]}},
                        title={"text": f"Estimated Fraud Probability ({assessment['model_used']})", "font": {"size": 15}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": assessment["color"], "thickness": 0.25},
                            "steps": [
                                {"range": [0, low_thresh * 100], "color": "#dcfce7"},
                                {"range": [low_thresh * 100, high_thresh * 100], "color": "#fef3c7"},
                                {"range": [high_thresh * 100, 100], "color": "#fee2e2"}
                            ],
                            "threshold": {"line": {"color": "#dc2626", "width": 4}, "thickness": 0.75, "value": high_thresh * 100}
                        }
                    ))
                    fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_g, use_container_width=True)

                    st.markdown(f"""
                    <div style="text-align: center; margin-top: -10px;">
                        <span class="badge-{assessment['badge_style']}">
                            CATEGORY: {assessment['risk_level'].upper()}
                        </span>
                        <p style="margin-top: 8px; color: #475569; font-size: 0.9rem;">
                            {assessment['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    st.markdown("#### 🚨 Heuristic Red Flags & Rule Triggers")
                    flags = assessment["heuristic_flags"]
                    if flags:
                        for f in flags:
                            c = "#ef4444" if f["severity"] == "HIGH" else "#f59e0b" if f["severity"] == "MEDIUM" else "#64748b"
                            st.markdown(f"""
                            <div class="flag-card" style="border-left-color: {c};">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>{f['category']}</strong>
                                    <span style="font-size:0.75rem; font-weight:700; color:{c};">{f['severity']} SEVERITY</span>
                                </div>
                                <div style="font-size:0.85rem; color:#475569; margin-top:4px;">{f['description']}</div>
                                <div style="margin-top:4px;"><code style="font-size:0.75rem; background:#e2e8f0; padding:2px 6px; border-radius:4px;">Triggers: {', '.join(f['evidence'])}</code></div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No heuristic red flags detected in payment instructions, communication channels, or logo presence.")

                # Highlighted text
                st.markdown("#### 📝 Token Highlighting (Linguistic Explanations)")
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; line-height: 1.7; max-height: 220px; overflow-y: auto;">
                    {assessment['highlighted_html']}
                </div>
                """, unsafe_allow_html=True)
                st.caption("🔴 Red highlights: Known scam vocabulary or suspicious payment requests. 🟢 Green: Common markers of verified legitimate employment.")

                # Recommendations
                r_c1, r_c2 = st.columns([1.2, 1])
                with r_c1:
                    st.markdown("#### 🛡️ Safe Verification Checklist for Applicants")
                    for rec in assessment["recommendations"]:
                        st.markdown(f"- {rec}")
                with r_c2:
                    st.markdown("#### ⚠️ Responsible Use Notice")
                    st.markdown(f"""
                    <div class="disclaimer-card">
                        {assessment['disclaimer']}
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: DATASET EXPLORER & EDA
    # =========================================================================
    with tab_eda:
        st.markdown("### 📊 Exploratory Data Analysis: EMSCAD Benchmark Dataset")
        if raw_df is not None:
            stats = get_dataset_overview_stats(raw_df, file_size)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Postings", f"{stats['total_rows']:,}")
            col2.metric("Features", f"{stats['total_cols']}")
            col3.metric("Memory Footprint", f"{stats['memory_mb']} MB")
            col4.metric("Exact Duplicates", f"{stats['exact_duplicates']} ({stats['duplicate_pct']}%)")
            col5.metric("File Size on Disk", f"{stats['file_size_mb']} MB")

            # Missing values and Target distribution
            st.markdown("---")
            e_c1, e_c2 = st.columns(2)
            with e_c1:
                fig_bar, fig_donut = plot_target_distribution(raw_df, "fraudulent")
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
            with e_c2:
                if fig_donut:
                    st.plotly_chart(fig_donut, use_container_width=True)

            # Missingness
            st.markdown("---")
            st.markdown("#### 📉 Missing Values Analysis")
            m_bar, m_heat = plot_missing_values(raw_df)
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                if m_bar:
                    st.plotly_chart(m_bar, use_container_width=True)
            with m_c2:
                if m_heat:
                    st.plotly_chart(m_heat, use_container_width=True)

            # Metadata Correlation Heatmap
            st.markdown("---")
            st.markdown("#### 🔗 Metadata Pearson Correlation Heatmap")
            corr_fig = plot_metadata_correlation(raw_df, target_col="fraudulent")
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            st.caption("Focuses strictly on numeric metadata attributes and text length metrics; sparse TF-IDF columns are deliberately excluded to preserve interpretability.")

            # Categorical Distributions
            st.markdown("---")
            st.markdown("#### 🏢 Categorical Attributes vs Fraud Rate")
            cat_choice = st.selectbox("Select Categorical Attribute to Inspect:", ["employment_type", "required_experience", "required_education"])
            cat_fig = plot_categorical_feature_by_target(raw_df, cat_choice, "fraudulent")
            if cat_fig:
                st.plotly_chart(cat_fig, use_container_width=True)

            # Text Length Boxplots
            st.markdown("---")
            st.markdown("#### 📏 Text Length Distributions (Character Counts)")
            len_figs = plot_text_length_distributions(raw_df, "fraudulent")
            l_c1, l_c2 = st.columns(2)
            with l_c1:
                if "description" in len_figs:
                    st.plotly_chart(len_figs["description"], use_container_width=True)
            with l_c2:
                if "company_profile" in len_figs:
                    st.plotly_chart(len_figs["company_profile"], use_container_width=True)

            # N-Grams
            st.markdown("---")
            st.markdown("#### 🔤 Most Frequent N-Grams by Class")
            ng_choice = st.radio("Select N-Gram Range:", ["Unigrams (1-Word)", "Bigrams (2-Words)"], horizontal=True)
            ngram_tuple = (1, 1) if "Unigrams" in ng_choice else (2, 2)
            ng_legit, ng_fraud = compute_top_ngrams_by_class(raw_df, "description", "fraudulent", ngram_range=ngram_tuple, top_n=15)
            ng_c1, ng_c2 = st.columns(2)
            with ng_c1:
                if ng_legit:
                    st.plotly_chart(ng_legit, use_container_width=True)
            with ng_c2:
                if ng_fraud:
                    st.plotly_chart(ng_fraud, use_container_width=True)

            # Interactive Filtering
            st.markdown("---")
            st.markdown("#### 🔎 Interactive Dataset Filter")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                filter_target = st.multiselect("Filter by Target Label:", [0, 1], default=[0, 1], format_func=lambda x: "Legitimate (0)" if x == 0 else "Fraudulent (1)")
            with f_col2:
                exp_vals = list(raw_df["required_experience"].dropna().unique())
                filter_exp = st.multiselect("Filter by Experience:", exp_vals, default=exp_vals[:3])

            filtered_df = raw_df[raw_df["fraudulent"].isin(filter_target)]
            if filter_exp:
                filtered_df = filtered_df[filtered_df["required_experience"].isin(filter_exp)]

            st.write(f"Showing **{len(filtered_df):,}** of **{len(raw_df):,}** postings after filtering.")
            st.dataframe(filtered_df[["title", "company_profile", "telecommuting", "has_company_logo", "fraudulent"]].head(20), use_container_width=True)

        else:
            st.info("Raw dataset not found at `data/raw/fake_job_postings.csv`. Run `python train_evaluate.py` to download.")

    # =========================================================================
    # TAB 4: UPLOAD YOUR DATASET
    # =========================================================================
    with tab_upload:
        st.markdown("### 📂 Upload and Analyze Your Own Dataset (CSV)")
        st.write("Upload an external recruitment dataset to inspect its schema, run descriptive EDA, clean anomalies, and evaluate independently.")

        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Supported format: CSV (Max size: 50MB)")

        if uploaded_file is not None:
            try:
                # Safe read with encoding fallback
                try:
                    df_up = pd.read_csv(uploaded_file, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_up = pd.read_csv(uploaded_file, encoding="latin-1")

                st.success(f"Successfully loaded file: **{uploaded_file.name}** ({len(df_up):,} rows, {len(df_up.columns)} columns)")

                up_stats = get_dataset_overview_stats(df_up, uploaded_file.size)

                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Uploaded Rows", f"{up_stats['total_rows']:,}")
                u2.metric("Uploaded Columns", f"{up_stats['total_cols']}")
                u3.metric("Memory Usage", f"{up_stats['memory_mb']} MB")
                u4.metric("Exact Duplicates", f"{up_stats['exact_duplicates']}")

                # Column & Target Configuration
                st.markdown("---")
                st.markdown("#### 🎯 Target Column & Label Mapping")
                all_cols = list(df_up.columns)
                default_target = "fraudulent" if "fraudulent" in all_cols else all_cols[-1]
                target_choice = st.selectbox("Select Target Label Column:", all_cols, index=all_cols.index(default_target))

                # Inspect unique values in selected target
                unique_labels = list(df_up[target_choice].dropna().unique())
                st.write(f"Unique values detected in **{target_choice}**: `{unique_labels}`")

                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    fraud_val = st.selectbox("Which value represents Fraudulent / Positive class?", unique_labels, index=min(1, len(unique_labels)-1))
                with m_c2:
                    legit_val = st.selectbox("Which value represents Legitimate / Negative class?", unique_labels, index=0)

                # Map to binary 0/1 for independent analysis
                df_mapped = df_up.copy()
                df_mapped["_target_binary"] = df_mapped[target_choice].apply(lambda x: 1 if x == fraud_val else 0)

                st.markdown("---")
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    analyze_btn = st.button("📊 Run Independent EDA on Uploaded CSV", type="primary", use_container_width=True)
                with act_col2:
                    st.button("⚙️ Prepare and Train Model (Optional)", disabled=True, help="Training on custom uploads is decoupled to prevent overwriting project baseline artifacts.")

                if analyze_btn or st.session_state.get("up_analyzed", False):
                    st.session_state["up_analyzed"] = True
                    st.markdown("### 📈 Uploaded Dataset EDA Results")

                    up_bar, up_donut = plot_target_distribution(df_mapped, "_target_binary")
                    u_c1, u_c2 = st.columns(2)
                    with u_c1:
                        if up_bar:
                            st.plotly_chart(up_bar, use_container_width=True)
                    with u_c2:
                        if up_donut:
                            st.plotly_chart(up_donut, use_container_width=True)

                    up_m_bar, _ = plot_missing_values(df_up)
                    if up_m_bar:
                        st.plotly_chart(up_m_bar, use_container_width=True)

                # Cleaning Options
                st.markdown("---")
                st.markdown("#### 🧹 Interactive Data Cleaning Options")
                clean_dup = st.checkbox("Remove exact duplicate rows")
                clean_strip = st.checkbox("Trim leading/trailing whitespace across string columns", value=True)

                df_cleaned_export = df_up.copy()
                if clean_dup:
                    init_len = len(df_cleaned_export)
                    df_cleaned_export = df_cleaned_export.drop_duplicates()
                    st.info(f"Removed **{init_len - len(df_cleaned_export):,}** duplicate rows.")

                if clean_strip:
                    for c in df_cleaned_export.select_dtypes(include="object").columns:
                        df_cleaned_export[c] = df_cleaned_export[c].str.strip()

                # Download Cleaned Dataset
                csv_buffer = io.BytesIO()
                df_cleaned_export.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Download Cleaned CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"cleaned_{uploaded_file.name}",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

        else:
            st.info("Please upload a CSV file above to begin interactive exploration.")

    # =========================================================================
    # TAB 5: MODEL BENCHMARKS
    # =========================================================================
    with tab_benchmarks:
        st.markdown("### 📈 Comprehensive Model Comparison (Held-Out Test Set: 2,682 Listings)")
        st.write("All metrics are computed on the unbiased test partition with strict zero-leakage isolation.")

        if comp_df is not None:
            st.dataframe(
                comp_df.style.highlight_max(subset=["Fraud Precision", "Fraud Recall", "Fraud F1-Score", "PR-AUC", "ROC-AUC"], color="#bbf7d0"),
                use_container_width=True
            )

        if metrics and "models" in metrics:
            m_c1, m_c2 = st.columns(2)

            with m_c1:
                st.markdown("#### 📈 Precision-Recall Curves")
                fig_pr = go.Figure()
                for name, d in metrics["models"].items():
                    if d.get("pr_curve"):
                        pr = d["pr_curve"]
                        fig_pr.add_trace(go.Scatter(
                            x=pr["recall"],
                            y=pr["precision"],
                            mode="lines",
                            name=f"{name} (AUC={pr['pr_auc']:.3f})"
                        ))
                fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=380, hovermode="x unified")
                st.plotly_chart(fig_pr, use_container_width=True)

            with m_c2:
                st.markdown("#### 📈 ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
                for name, d in metrics["models"].items():
                    if d.get("roc_curve"):
                        roc = d["roc_curve"]
                        fig_roc.add_trace(go.Scatter(
                            x=roc["fpr"],
                            y=roc["tpr"],
                            mode="lines",
                            name=f"{name} (AUC={roc['roc_auc']:.3f})"
                        ))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380, hovermode="x unified")
                st.plotly_chart(fig_roc, use_container_width=True)

            # Confusion Matrix Inspector
            st.markdown("---")
            st.markdown("#### 🔲 Confusion Matrix Breakdown")
            selected_cm_model = st.selectbox("Inspect Confusion Matrix for Model:", list(metrics["models"].keys()), index=2)
            cm_info = metrics["models"][selected_cm_model]["test_metrics"]["confusion_matrix"]

            matrix_vals = [
                [cm_info["tn"], cm_info["fp"]],
                [cm_info["fn"], cm_info["tp"]]
            ]

            fig_cm = px.imshow(
                matrix_vals,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Legitimate (0)", "Fraudulent (1)"],
                y=["Legitimate (0)", "Fraudulent (1)"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=320, width=500)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.caption(f"**Counts:** True Negatives: {cm_info['tn']} ({cm_info.get('tn_pct', 0)}%) | False Positives: {cm_info['fp']} ({cm_info.get('fp_pct', 0)}%) | False Negatives: {cm_info['fn']} ({cm_info.get('fn_pct', 0)}%) | True Positives: {cm_info['tp']} ({cm_info.get('tp_pct', 0)}%).")

        else:
            st.info("Metrics not loaded. Execute `python train_evaluate.py` to populate benchmarks.")

    # =========================================================================
    # TAB 6: EXPLAINABILITY & ERROR ANALYSIS
    # =========================================================================
    with tab_explain:
        st.markdown("### 🔬 Explainability & Error Analysis")

        # Global feature importance
        if metrics and "global_explainability" in metrics:
            st.markdown("#### 🔤 Most Influential Learned Associations (Logistic Regression Weights)")
            st.caption("Feature weights reflect statistical correlations in the training data; they do not establish causal proof of fraud.")

            ge = metrics["global_explainability"]
            top_fraud = pd.DataFrame(ge["top_fraud_terms"][:12], columns=["Term", "Weight"])
            top_legit = pd.DataFrame(ge["top_legit_terms"][:12], columns=["Term", "Weight"])

            ex_c1, ex_c2 = st.columns(2)
            with ex_c1:
                fig_f = px.bar(top_fraud, x="Weight", y="Term", orientation="h", title="Top Scam-Associated Features", color_discrete_sequence=["#ef4444"])
                fig_f.update_layout(yaxis=dict(autorange="reversed"), height=380)
                st.plotly_chart(fig_f, use_container_width=True)

            with ex_c2:
                fig_l = px.bar(top_legit, x="Weight", y="Term", orientation="h", title="Top Legitimate-Associated Features", color_discrete_sequence=["#10b981"])
                fig_l.update_layout(yaxis=dict(autorange="reversed"), height=380)
                st.plotly_chart(fig_l, use_container_width=True)

        # Feature Ablation Study
        if metrics and "ablation_study" in metrics:
            st.markdown("---")
            st.markdown("#### 🔬 Feature Ablation: Value of Structured Metadata")
            ab = metrics["ablation_study"]
            to = ab["text_only"]
            cb = ab["combined"]

            ab_df = pd.DataFrame([
                {"Feature Set": "Text Only (TF-IDF alone)", "Precision": to["fraud_precision"], "Recall": to["fraud_recall"], "F1-Score": to["fraud_f1"], "PR-AUC": to["pr_auc"]},
                {"Feature Set": "Combined (Text + Metadata)", "Precision": cb["fraud_precision"], "Recall": cb["fraud_recall"], "F1-Score": cb["fraud_f1"], "PR-AUC": cb["pr_auc"]}
            ])
            st.dataframe(ab_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "PR-AUC": "{:.4f}"}), use_container_width=True)
            st.success(f"Adding structured metadata improved PR-AUC by **{ab.get('delta_pr_auc', 0):+.4f}** and caught **{ab.get('delta_recall', 0)*100:+.2f}%** more scams!")

        # In-Depth Error Analysis
        if metrics and "error_analysis" in metrics:
            st.markdown("---")
            st.markdown("#### 🔍 Error Analysis: Understanding Model Mistakes")
            ea = metrics["error_analysis"]

            e_tab1, e_tab2, e_tab3 = st.tabs(["Typologies of Failure", "False Positives (False Alarms)", "False Negatives (Missed Scams)"])

            with e_tab1:
                for typ in ea["error_typologies"]:
                    st.markdown(f"**• {typ['category']}:** {typ['impact']}")

            with e_tab2:
                st.write("Examples where legitimate job advertisements were incorrectly classified as fraudulent:")
                for ex in ea["false_positive_examples"]:
                    st.markdown(f"""
                    <div style="background:#f8fafc; border-left:3px solid #f59e0b; padding:10px 14px; margin-bottom:8px; border-radius:4px;">
                        <strong>Job #{ex['job_id']}: {ex['title']}</strong> (Predicted Prob: {ex['predicted_prob']:.2f})<br>
                        <span style="font-size:0.85rem; color:#64748b;">Cause: {ex['reason_category']}</span><br>
                        <code style="font-size:0.8rem;">Snippet: {ex['snippet']}</code>
                    </div>
                    """, unsafe_allow_html=True)

            with e_tab3:
                st.write("Examples where actual fraudulent listings evaded detection:")
                for ex in ea["false_negative_examples"]:
                    st.markdown(f"""
                    <div style="background:#f8fafc; border-left:3px solid #ef4444; padding:10px 14px; margin-bottom:8px; border-radius:4px;">
                        <strong>Job #{ex['job_id']}: {ex['title']}</strong> (Predicted Prob: {ex['predicted_prob']:.2f})<br>
                        <span style="font-size:0.85rem; color:#64748b;">Cause: {ex['reason_category']}</span><br>
                        <code style="font-size:0.8rem;">Snippet: {ex['snippet']}</code>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 7: METHODOLOGY & LIMITATIONS
    # =========================================================================
    with tab_methodology:
        st.markdown("### 📖 Methodology, Leakage Prevention, & Viva Guide")

        viva_tab1, viva_tab2, viva_tab3 = st.tabs(["Leakage Prevention Protocol", "Class Imbalance Strategy", "🎓 Top 5 Viva Questions & Answers"])

        with viva_tab1:
            st.markdown("""
            #### How Data Leakage is Strictly Prevented:
            1. **Temporal / Stratified Split Isolation**: The raw dataset was partitioned into Train (70%), Validation (15%), and Test (15%) splits *before* any feature extraction occurred.
            2. **Fitted strictly on Train**: `TfidfVectorizer` (sublinear TF, stopwords, max 10k n-grams) and `OneHotEncoder` were fitted **strictly on the 12,516 training samples**. Validation and test data were transformed using the frozen preprocessor.
            3. **Identifier Exclusion**: `job_id` and raw labels were explicitly isolated from the feature matrix to prevent identifier memorization.
            4. **Near-Duplicate Evaluation**: The dataset contains 3,078 repeated descriptions across staffing agencies. In production, group-aware deduplication is evaluated to prevent memorizing corporate templates across partitions.
            """)

        with viva_tab2:
            st.markdown("""
            #### Imbalanced Classification Methodology:
            - **Prevalence**: Only **4.84%** of listings are fraudulent (~19:1 ratio).
            - **Accuracy Paradox**: A naive model predicting all listings as legitimate achieves **95.16% accuracy** while catching zero scams!
            - **Algorithmic Mitigation**: `class_weight='balanced'` in Logistic Regression, Linear SVM, and Random Forest inversely weights loss by class frequency.
            - **Evaluation Metrics**: Models are ranked on **PR-AUC (Average Precision)**, **Fraud Precision**, **Fraud Recall**, and **Fraud F1-Score**.
            """)

        with viva_tab3:
            st.markdown("""
            #### 🎓 Essential Viva Questions & Answers for Your Professor:
            
            **Q1: Why did you choose Linear SVM and Logistic Regression over complex deep learning?**  
            *A1:* In high-dimensional, sparse text spaces (10,000 TF-IDF features), linear decision hyperplanes perform exceptionally well with convex optimization, require minimal compute, and provide direct coefficient interpretability (log-odds), which is essential for an explainable screening tool.
            
            **Q2: Why is PR-AUC preferred over ROC-AUC for this problem?**  
            *A2:* In severe class imbalance (4.8% fraud), ROC-AUC can remain deceptively high (e.g. 0.99) because the vast number of True Negatives suppresses the False Positive Rate. PR-AUC focuses strictly on the minority fraud class, reflecting genuine precision across all recall thresholds.
            
            **Q3: How did you prove that structured metadata actually helps?**  
            *A3:* We executed a controlled Feature Ablation experiment. Training Logistic Regression on text-only achieved a PR-AUC of 0.8999, whereas adding structured metadata increased PR-AUC to 0.9116 and boosted recall, demonstrating that attributes like missing corporate logos and absent profiles provide strong discriminative regularization.
            
            **Q4: How do you handle probability calibration in Linear SVM?**  
            *A4:* Standard Linear SVM outputs uncalibrated decision distances, not probabilities. We applied Platt scaling via scikit-learn's `CalibratedClassifierCV(method='sigmoid', cv=3)`, which fits a logistic regression model over the SVM decision values to output valid posterior probabilities.
            
            **Q5: What are the primary cybersecurity limitations of this project?**  
            *A5:* The EMSCAD dataset is historical (classical check cashing, wire transfers, data entry). Modern threat actors leverage Generative AI (LLMs) to write flawless corporate prose, evading static keyword filters. Therefore, this tool is strictly a preliminary screening aid, not definitive employer verification.
            """)


if __name__ == "__main__":
    main()
