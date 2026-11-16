# Main Streamlit application for the AWS Healthcare Analytics portfolio.
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Healthcare Analytics Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare Analytics Intelligence")
st.caption("AWS Glue | PySpark | Machine Learning | Amazon Athena | Streamlit")

st.markdown("""
Interactive healthcare analytics platform built with synthetic
patient-level data, longitudinal disease cohorts, and statistical models.
""")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a dashboard",
    [
        "Executive Overview",
        "Chronic Disease Cohorts",
        "Patient Journeys",
        "Statistical Modeling",
        "Provider Entity Resolution"
    ]
)

st.subheader(page)

# Executive overview of the chronic disease analytics portfolio.
if page == "Executive Overview":
    data_path = Path(__file__).parent / "data" / "chronic_disease_cohort_summary"
    files = list(data_path.glob("*.parquet"))

    if not files:
        st.error("Cohort data not found.")
        st.stop()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Calculate portfolio-level cohort metrics.
    total_patients = int(df["patient_count"].sum())
    complete_windows = int(df["patients_with_both_calendar_windows"].sum())
    multimorbid = int(
        df.loc[df["disease_count"] >= 2, "patient_count"].sum()
    )
    multimorbidity_pct = 100 * multimorbid / total_patients if total_patients else 0

    # Display executive KPI cards.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients in Cohorts", f"{total_patients:,}")
    c2.metric("Multimorbid Patients", f"{multimorbid:,}")
    c3.metric("Multimorbidity Rate", f"{multimorbidity_pct:.1f}%")
    c4.metric("Complete Calendar Windows", f"{complete_windows:,}")

    # Visualize chronic disease burden.
    df_burden = df.groupby("disease_count", as_index=False)["patient_count"].sum()

    fig_burden = px.bar(
        df_burden,
        x="disease_count",
        y="patient_count",
        text="patient_count",
        title="Patient Distribution by Number of Chronic Conditions",
        labels={
            "disease_count": "Number of Chronic Conditions",
            "patient_count": "Patients"
        }
    )
    fig_burden.update_xaxes(type="category")
    st.plotly_chart(fig_burden, use_container_width=True)

    st.caption(
        "Source: AWS Glue curated chronic disease cohorts. "
        "All patient records are synthetic Synthea data."
    )

    

# Display actual chronic disease cohort data from the AWS Glue pipeline.
if page == "Chronic Disease Cohorts":
    data_path = Path(__file__).parent / "data" / "chronic_disease_cohort_summary"
    files = list(data_path.glob("*.parquet"))

    if not files:
        st.error("Cohort data not found. Download the Parquet files from S3.")
        st.stop()

    # Load and combine all Parquet partitions.
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Interactive cohort selection.
    selected = st.multiselect(
        "Select chronic disease cohorts",
        options=df["cohort_segment"].tolist(),
        default=df["cohort_segment"].tolist()
    )
    filtered = df[df["cohort_segment"].isin(selected)]

    # Display total patients and cohort distribution.
    st.metric("Patients across selected cohorts", f"{filtered['patient_count'].sum():,}")

    fig = px.bar(
        filtered,
        x="cohort_segment",
        y="patient_count",
        color="disease_count",
        title="Patient Distribution by Chronic Disease Cohort",
        labels={"cohort_segment": "Disease Cohort", "patient_count": "Patients"}
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # Compare healthcare utilization and costs before and after diagnosis.
st.subheader("Healthcare Utilization and Costs")

metric_options = {
    "Average encounters": (
        "complete_window_avg_pre_encounters_180d",
        "complete_window_avg_post_encounters_180d"
    ),
    "Average claim costs ($)": (
        "complete_window_avg_pre_claim_cost_180d",
        "complete_window_avg_post_claim_cost_180d"
    )
}

selected_metric = st.selectbox(
    "Select metric",
    options=list(metric_options)
)

before_col, after_col = metric_options[selected_metric]

# Convert wide columns into a format suitable for grouped bars.
df_compare = filtered[
    ["cohort_segment", before_col, after_col]
].melt(
    id_vars="cohort_segment",
    var_name="Period",
    value_name="Value"
)

df_compare["Period"] = df_compare["Period"].map({
    before_col: "Before diagnosis",
    after_col: "After diagnosis"
})

# Create an interactive before/after comparison.
fig_compare = px.bar(
    df_compare,
    x="cohort_segment",
    y="Value",
    color="Period",
    barmode="group",
    title=f"{selected_metric}: 180 Days Before vs. After Diagnosis",
    labels={"cohort_segment": "Disease Cohort"}
)

fig_compare.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig_compare, use_container_width=True)