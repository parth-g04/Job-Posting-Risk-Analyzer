"""
Exploratory Data Analysis (EDA) module.
Provides comprehensive descriptive statistics, distribution calculations,
and interactive Plotly visualizations for default and uploaded datasets.
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_dataset_overview_stats(df: pd.DataFrame, file_size_bytes: int = None) -> dict:
    """Compute high-level dataset metrics, missingness, and duplicate summaries."""
    total_rows, total_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    exact_duplicates = int(df.duplicated().sum())
    duplicate_pct = (exact_duplicates / total_rows * 100) if total_rows > 0 else 0.0

    missing_series = df.isnull().sum()
    missing_pct_series = (missing_series / total_rows * 100) if total_rows > 0 else missing_series

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Missing Count": missing_series.values,
        "Missing %": missing_pct_series.round(2).values,
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns]
    }).sort_values(by="Missing Count", ascending=False)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_mb": round(memory_mb, 2),
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else None,
        "exact_duplicates": exact_duplicates,
        "duplicate_pct": round(duplicate_pct, 2),
        "missing_summary_df": missing_df
    }


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """Plot target class distribution and class imbalance ratios."""
    if target_col not in df.columns:
        return None, None

    counts = df[target_col].value_counts(dropna=False).reset_index()
    counts.columns = [target_col, "Count"]
    total = len(df)
    counts["Percentage"] = (counts["Count"] / total * 100).round(2)

    # Human-readable labels
    label_map = {0: "Legitimate (0)", 1: "Fraudulent (1)"}
    counts["Label_Name"] = counts[target_col].map(lambda x: label_map.get(x, str(x)))

    # Bar chart
    fig_bar = px.bar(
        counts,
        x="Label_Name",
        y="Count",
        text="Count",
        color="Label_Name",
        color_discrete_map={"Legitimate (0)": "#10b981", "Fraudulent (1)": "#ef4444"},
        title=f"Target Class Distribution: {target_col}"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, height=350, yaxis_title="Total Postings")

    # Donut chart
    fig_donut = px.pie(
        counts,
        names="Label_Name",
        values="Count",
        hole=0.45,
        color="Label_Name",
        color_discrete_map={"Legitimate (0)": "#10b981", "Fraudulent (1)": "#ef4444"},
        title="Class Proportion (Imbalance Ratio)"
    )
    fig_donut.update_traces(textinfo="percent+label")
    fig_donut.update_layout(height=350)

    return fig_bar, fig_donut


def plot_missing_values(df: pd.DataFrame):
    """Plot missingness percentage bar chart and heatmap across columns."""
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=True)

    if missing_cols.empty:
        return None, None

    # Bar chart
    pct_series = (missing_cols / len(df) * 100).round(2)
    fig_bar = px.bar(
        x=pct_series.values,
        y=pct_series.index,
        orientation="h",
        text=pct_series.values,
        labels={"x": "Missing Percentage (%)", "y": "Feature Name"},
        title="Missing Values by Feature (%)",
        color=pct_series.values,
        color_continuous_scale="Reds"
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(height=max(350, len(pct_series) * 22), coloraxis_showscale=False)

    # Sample heatmap for performance responsiveness
    sample_size = min(300, len(df))
    sample_df = df[missing_cols.index].sample(n=sample_size, random_state=42) if len(df) > sample_size else df[missing_cols.index]
    null_matrix = sample_df.isnull().astype(int).values

    fig_heatmap = px.imshow(
        null_matrix,
        labels=dict(x="Feature", y="Sampled Rows", color="Missing (1)"),
        x=list(missing_cols.index),
        color_continuous_scale=[[0, "#f8fafc"], [1, "#ef4444"]],
        title=f"Missingness Matrix (Sample of {sample_size} Postings)"
    )
    fig_heatmap.update_layout(height=350)

    return fig_bar, fig_heatmap


def plot_metadata_correlation(df: pd.DataFrame, target_col: str = None):
    """
    Correlation heatmap for numeric and structured metadata attributes.
    Excludes identifiers (job_id) and raw sparse text.
    """
    candidate_cols = [
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "has_company_profile",
        "salary_specified",
        "title_length",
        "description_length"
    ]

    # Dynamically engineer length features if not present
    df_eval = df.copy()
    if "description" in df_eval.columns and "description_length" not in df_eval.columns:
        df_eval["description_length"] = df_eval["description"].fillna("").astype(str).str.len()
    if "title" in df_eval.columns and "title_length" not in df_eval.columns:
        df_eval["title_length"] = df_eval["title"].fillna("").astype(str).str.len()
    if "company_profile" in df_eval.columns and "has_company_profile" not in df_eval.columns:
        df_eval["has_company_profile"] = (df_eval["company_profile"].fillna("").astype(str).str.len() > 10).astype(int)

    selected_cols = [c for c in candidate_cols if c in df_eval.columns]

    if target_col and target_col in df_eval.columns:
        selected_cols.append(target_col)

    if len(selected_cols) < 2:
        return None

    corr_df = df_eval[selected_cols].corr(method="pearson").round(3)

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Metadata Pearson Correlation Matrix"
    )
    fig.update_layout(height=420)
    return fig


def plot_categorical_feature_by_target(df: pd.DataFrame, feature_col: str, target_col: str = "fraudulent"):
    """Compare categorical feature frequency and fraud prevalence rate."""
    if feature_col not in df.columns or target_col not in df.columns:
        return None

    df_clean = df[[feature_col, target_col]].copy()
    df_clean[feature_col] = df_clean[feature_col].fillna("Unknown").astype(str)

    # Limit to top 10 categories for readability
    top_cats = df_clean[feature_col].value_counts().head(10).index
    df_filtered = df_clean[df_clean[feature_col].isin(top_cats)]

    grouped = df_filtered.groupby(feature_col)[target_col].agg(
        Total="count",
        Fraud_Count="sum",
        Fraud_Rate=lambda x: (x.sum() / x.count() * 100)
    ).reset_index().sort_values(by="Total", ascending=False)

    fig = px.bar(
        grouped,
        x=feature_col,
        y="Total",
        color="Fraud_Rate",
        text=grouped["Fraud_Rate"].apply(lambda x: f"{x:.1f}% Fraud"),
        labels={"Total": "Posting Volume", "Fraud_Rate": "Fraud Rate (%)"},
        title=f"Category Distribution & Fraud Rate: {feature_col}",
        color_continuous_scale="Viridis"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, xaxis_tickangle=-30)
    return fig


def plot_text_length_distributions(df: pd.DataFrame, target_col: str = "fraudulent"):
    """Plot text length distributions comparison between legitimate and fraudulent postings."""
    text_fields = ["description", "requirements", "company_profile", "title"]
    figures = {}

    df_eval = df.copy()
    has_target = target_col in df_eval.columns

    for field in text_fields:
        if field in df_eval.columns:
            len_col = f"{field}_char_len"
            df_eval[len_col] = df_eval[field].fillna("").astype(str).str.len()

            # Filter out extreme outliers for visualization clarity (e.g. 99th percentile)
            upper_bound = df_eval[len_col].quantile(0.99)
            df_plot = df_eval[df_eval[len_col] <= upper_bound]

            if has_target:
                df_plot["Label"] = df_plot[target_col].map({0: "Legitimate (0)", 1: "Fraudulent (1)"})
                fig = px.box(
                    df_plot,
                    x="Label",
                    y=len_col,
                    color="Label",
                    color_discrete_map={"Legitimate (0)": "#10b981", "Fraudulent (1)": "#ef4444"},
                    title=f"Character Length: {field.replace('_', ' ').title()}",
                    points=False
                )
            else:
                fig = px.histogram(
                    df_plot,
                    x=len_col,
                    nbins=40,
                    title=f"Character Length: {field.replace('_', ' ').title()}"
                )
            fig.update_layout(height=300, showlegend=False)
            figures[field] = fig

    return figures


def compute_top_ngrams_by_class(
    df: pd.DataFrame,
    text_col: str,
    target_col: str = "fraudulent",
    ngram_range=(1, 1),
    top_n: int = 15
):
    """Extract top unigrams or bigrams split by legitimate vs fraudulent class."""
    if text_col not in df.columns or target_col not in df.columns:
        return None, None

    # Custom stop words
    stop_words = set([
        "the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this",
        "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your",
        "all", "have", "new", "more", "an", "was", "we", "will", "home", "can", "us",
        "our", "work", "job", "experience", "skills", "team", "company"
    ])

    def tokenize_ngrams(series):
        tokens = []
        for text in series.dropna().astype(str):
            clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
            words = [w for w in clean.split() if len(w) > 2 and w not in stop_words]
            if ngram_range == (1, 1):
                tokens.extend(words)
            elif ngram_range == (2, 2):
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                tokens.extend(bigrams)
        return Counter(tokens)

    legit_counts = tokenize_ngrams(df[df[target_col] == 0][text_col])
    fraud_counts = tokenize_ngrams(df[df[target_col] == 1][text_col])

    df_legit = pd.DataFrame(legit_counts.most_common(top_n), columns=["N-Gram", "Frequency"])
    df_fraud = pd.DataFrame(fraud_counts.most_common(top_n), columns=["N-Gram", "Frequency"])

    ngram_label = "Unigrams" if ngram_range == (1, 1) else "Bigrams"

    fig_legit = px.bar(
        df_legit,
        x="Frequency",
        y="N-Gram",
        orientation="h",
        title=f"Top {ngram_label} — Legitimate Listings",
        color_discrete_sequence=["#10b981"]
    )
    fig_legit.update_layout(yaxis=dict(autorange="reversed"), height=380)

    fig_fraud = px.bar(
        df_fraud,
        x="Frequency",
        y="N-Gram",
        orientation="h",
        title=f"Top {ngram_label} — Fraudulent Listings",
        color_discrete_sequence=["#ef4444"]
    )
    fig_fraud.update_layout(yaxis=dict(autorange="reversed"), height=380)

    return fig_legit, fig_fraud
