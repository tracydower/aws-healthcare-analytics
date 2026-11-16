# Public Streamlit dashboard for curated AWS healthcare analytics datasets.
import io

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Healthcare Analytics Intelligence",
    page_icon="🏥",
    layout="wide",
)


# Read approved, synthetic cohort summary data from S3 and cache for one hour.
@st.cache_data(ttl=3600, show_spinner="Loading healthcare analytics data from AWS...")
def load_cohort_data():
    aws = st.secrets["aws"]
    s3 = boto3.client(
        "s3",
        region_name=aws["region"],
        aws_access_key_id=aws["access_key_id"],
        aws_secret_access_key=aws["secret_access_key"],
    )
    prefix = "curated/chronic_disease_cohort_summary/"
    frames = []
    paginator = s3.get_paginator("list_objects_v2")
    for response in paginator.paginate(Bucket=aws["bucket"], Prefix=prefix):
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                result = s3.get_object(Bucket=aws["bucket"], Key=obj["Key"])
                frames.append(pd.read_parquet(io.BytesIO(result["Body"].read())))
    if not frames:
        raise FileNotFoundError("No cohort summary Parquet files found in the configured S3 prefix.")
    return pd.concat(frames, ignore_index=True)


st.title("Healthcare Analytics Intelligence")
st.caption("AWS Glue | PySpark | Machine Learning | Amazon S3 | Streamlit")
st.markdown(
    "Interactive healthcare analytics platform built from synthetic patient-level "
    "data, longitudinal disease cohorts, and statistical models."
)
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a dashboard",
    [
        "Executive Overview",
        "Chronic Disease Cohorts",
        "Patient Journeys",
        "Statistical Modeling",
        "Provider Entity Resolution",
    ],
)
st.subheader(page)

if page in ("Executive Overview", "Chronic Disease Cohorts"):
    try:
        df = load_cohort_data()
    except Exception as exc:
        st.error("Could not load the cohort summary from AWS S3.")
        st.caption(f"Error type: {type(exc).__name__}. Check Streamlit secrets and the IAM user's S3 read permissions.")
        st.stop()

    if page == "Executive Overview":
        # Seven mutually exclusive cohort segments total 792 synthetic patients.
        total_patients = int(df["patient_count"].sum())
        complete_windows = int(df["patients_with_both_calendar_windows"].sum())
        multimorbid = int(df.loc[df["disease_count"] >= 2, "patient_count"].sum())
        multimorbidity_pct = 100 * multimorbid / total_patients if total_patients else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients in Cohorts", f"{total_patients:,}")
        c2.metric("Multimorbid Patients", f"{multimorbid:,}")
        c3.metric("Multimorbidity Rate", f"{multimorbidity_pct:.1f}%")
        c4.metric("Complete Calendar Windows", f"{complete_windows:,}")

        df_burden = df.groupby("disease_count", as_index=False)["patient_count"].sum()
        df_burden["condition_count"] = df_burden["disease_count"].astype(str)
        fig_burden = px.bar(
            df_burden,
            x="condition_count",
            y="patient_count",
            text="patient_count",
            title="Patient Distribution by Number of Chronic Conditions",
            labels={"condition_count": "Number of Chronic Conditions", "patient_count": "Patients"},
        )
        st.plotly_chart(fig_burden, use_container_width=True)
        st.caption(
            "Source: AWS Glue curated chronic disease cohorts. "
            "All patient records are synthetic Synthea data."
        )

    elif page == "Chronic Disease Cohorts":
        selected = st.multiselect(
            "Select chronic disease cohorts",
            options=df["cohort_segment"].tolist(),
            default=df["cohort_segment"].tolist(),
        )
        filtered = df[df["cohort_segment"].isin(selected)].copy()
        st.metric("Patients across selected cohorts", f"{filtered['patient_count'].sum():,.0f}")
        if filtered.empty:
            st.info("Select at least one cohort to display charts.")
            st.stop()

        filtered["disease_count_label"] = filtered["disease_count"].astype(str)
        fig = px.bar(
            filtered,
            x="cohort_segment",
            y="patient_count",
            color="disease_count_label",
            title="Patient Distribution by Chronic Disease Cohort",
            labels={
                "cohort_segment": "Disease Cohort",
                "patient_count": "Patients",
                "disease_count_label": "Number of Conditions",
            },
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        # Keep all before/after controls and charts inside the cohort page.
        st.subheader("Healthcare Utilization and Costs")
        metric_options = {
            "Average encounters": (
                "complete_window_avg_pre_encounters_180d",
                "complete_window_avg_post_encounters_180d",
            ),
            "Average claim costs ($)": (
                "complete_window_avg_pre_claim_cost_180d",
                "complete_window_avg_post_claim_cost_180d",
            ),
        }
        selected_metric = st.selectbox("Select metric", options=list(metric_options))
        before_col, after_col = metric_options[selected_metric]
        df_compare = filtered[["cohort_segment", before_col, after_col]].melt(
            id_vars="cohort_segment", var_name="Period", value_name="Value"
        )
        df_compare["Period"] = df_compare["Period"].map(
            {before_col: "Before diagnosis", after_col: "After diagnosis"}
        )
        fig_compare = px.bar(
            df_compare,
            x="cohort_segment",
            y="Value",
            color="Period",
            barmode="group",
            title=f"{selected_metric}: 180 Days Before vs. After Diagnosis",
            labels={"cohort_segment": "Disease Cohort"},
        )
        fig_compare.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption(
            "Descriptive comparisons among patients with complete 180-day "
            "calendar observation windows; differences are not causal effects."
        )
else:
    st.info("This dashboard section is under development. Curated AWS data will be connected next.")
