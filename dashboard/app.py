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


# Retrieve approved synthetic Parquet datasets from S3, caching each for one hour.
@st.cache_data(ttl=3600, show_spinner="Loading healthcare analytics data from AWS...")
def load_s3_dataset(prefix):
    aws = st.secrets["aws"]
    s3 = boto3.client(
        "s3",
        region_name=aws["region"],
        aws_access_key_id=aws["access_key_id"],
        aws_secret_access_key=aws["secret_access_key"],
    )
    frames = []
    paginator = s3.get_paginator("list_objects_v2")
    for response in paginator.paginate(Bucket=aws["bucket"], Prefix=prefix):
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                result = s3.get_object(Bucket=aws["bucket"], Key=obj["Key"])
                frames.append(pd.read_parquet(io.BytesIO(result["Body"].read())))
    if not frames:
        raise FileNotFoundError(f"No Parquet files found at s3://{aws['bucket']}/{prefix}")
    return pd.concat(frames, ignore_index=True)


def load_cohort_data():
    return load_s3_dataset("curated/chronic_disease_cohort_summary/")


def load_journey_data():
    return load_s3_dataset("curated/patient_disease_journeys/")


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
elif page == "Patient Journeys":
    try:
        journeys = load_journey_data()
    except Exception as exc:
        st.error("Could not load patient journeys from AWS S3.")
        st.code(str(exc))
        st.stop()

    # Filter at the dataset and disease-event level; never expose patient identifiers.
    sources = sorted(journeys["source_dataset"].dropna().astype(str).unique().tolist())
    chosen_sources = st.multiselect("Source datasets", sources, default=sources)
    journeys = journeys[journeys["source_dataset"].isin(chosen_sources)].copy()
    if journeys.empty:
        st.info("Select at least one source dataset.")
        st.stop()

    # A patient can have multiple disease events; counts here are event-level unless stated.
    event_labels = (
        journeys[["event_code", "diagnosis_description"]]
        .drop_duplicates(subset=["event_code"])
        .copy()
    )
    event_labels["diagnosis_description"] = event_labels["diagnosis_description"].fillna("Unspecified diagnosis")
    label_map = dict(zip(event_labels["event_code"], event_labels["diagnosis_description"]))
    event_codes = sorted(journeys["event_code"].dropna().unique().tolist(), key=str)
    chosen_codes = st.multiselect(
        "Diagnosis events",
        options=event_codes,
        default=event_codes,
        format_func=lambda code: f"{label_map.get(code, 'Diagnosis')} ({code})",
    )
    journeys = journeys[journeys["event_code"].isin(chosen_codes)].copy()
    if journeys.empty:
        st.info("Select at least one diagnosis event.")
        st.stop()

    journeys["index_diagnosis_date"] = pd.to_datetime(journeys["index_diagnosis_date"], errors="coerce")
    # Complete calendar windows are required for comparable 180-day before/after metrics.
    for column in ("has_180d_prior_recorded_history", "has_180d_calendar_followup"):
        journeys[column] = journeys[column].astype("string").str.lower().isin(["true", "1", "yes"])
    complete = journeys[
        journeys["has_180d_prior_recorded_history"] & journeys["has_180d_calendar_followup"]
    ].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disease-event records", f"{len(journeys):,}")
    c2.metric("Distinct patients", f"{journeys[['source_dataset', 'patient_id']].drop_duplicates().shape[0]:,}")
    c3.metric("Complete 180-day windows", f"{len(complete):,}")
    c4.metric("Diagnosis codes", f"{journeys['event_code'].nunique():,}")

    st.subheader("Diagnosis events over time")
    timeline = journeys.dropna(subset=["index_diagnosis_date"]).copy()
    timeline["diagnosis_year"] = timeline["index_diagnosis_date"].dt.year
    annual = timeline.groupby("diagnosis_year", as_index=False).size().rename(columns={"size": "event_records"})
    if not annual.empty:
        fig_timeline = px.bar(
            annual, x="diagnosis_year", y="event_records",
            title="Recorded first-diagnosis events by year",
            labels={"diagnosis_year": "Diagnosis year", "event_records": "Disease-event records"},
        )
        fig_timeline.update_xaxes(type="category")
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No valid diagnosis dates in the selection.")

    st.subheader("180-day utilization and cost comparison")
    st.caption(
        "Comparison uses only disease-event records with both recorded 180-day prior history "
        "and 180-day calendar follow-up. Results are descriptive, not causal."
    )
    metric_options = {
        "Encounters": ("pre_encounters_180d", "post_encounters_180d"),
        "Emergency encounters": ("pre_emergency_encounters_180d", "post_emergency_encounters_180d"),
        "Inpatient encounters": ("pre_inpatient_encounters_180d", "post_inpatient_encounters_180d"),
        "Outpatient encounters": ("pre_outpatient_encounters_180d", "post_outpatient_encounters_180d"),
        "Claims": ("pre_claims_180d", "post_claims_180d"),
        "Procedures": ("pre_procedures_180d", "post_procedures_180d"),
        "Encounter claim cost ($)": ("pre_encounter_claim_cost_180d", "post_encounter_claim_cost_180d"),
        "Payer coverage ($)": ("pre_payer_coverage_180d", "post_payer_coverage_180d"),
    }
    selected_metric = st.selectbox("Metric", list(metric_options), key="journey_metric")
    pre_col, post_col = metric_options[selected_metric]
    if complete.empty:
        st.warning("No records in this selection have both complete observation windows.")
    else:
        paired = complete[["event_code", pre_col, post_col]].copy()
        paired[pre_col] = pd.to_numeric(paired[pre_col], errors="coerce")
        paired[post_col] = pd.to_numeric(paired[post_col], errors="coerce")
        paired = paired.dropna(subset=[pre_col, post_col])
        if paired.empty:
            st.info("No paired values are available for this metric.")
        else:
            # Both averages use identical paired event records.
            means = paired[[pre_col, post_col]].mean()
            comparison = pd.DataFrame({
                "Period": ["180 days before", "180 days after"],
                "Average per disease-event record": [means[pre_col], means[post_col]],
            })
            chart = px.bar(
                comparison, x="Period", y="Average per disease-event record", color="Period",
                title=f"Average {selected_metric.lower()} by observation window",
            )
            st.plotly_chart(chart, use_container_width=True)
            st.caption(f"Paired records included: {len(paired):,}. Patients may contribute multiple disease-event records.")

    st.subheader("Most frequent diagnosis events")
    frequency = journeys.groupby(["event_code", "diagnosis_description"], dropna=False).size().reset_index(name="event_records")
    frequency = frequency.sort_values("event_records", ascending=False).head(15)
    st.dataframe(
        frequency.rename(columns={
            "event_code": "Diagnosis code",
            "diagnosis_description": "Diagnosis",
            "event_records": "Disease-event records",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("All data is synthetic. Individual patient identifiers are not displayed.")
elif page == "Statistical Modeling":
    st.caption(
        "Exploratory Poisson regression and segment-level comparisons from synthetic Synthea data. "
        "These results are associations, not treatment effects or clinical predictions."
    )
    try:
        coefficients = load_s3_dataset("curated/model_coefficients/")
        diagnostics = load_s3_dataset("curated/model_diagnostics/")
        segments = load_s3_dataset("curated/segment_summary/")
    except Exception as exc:
        st.error("Could not load statistical modeling results from AWS S3.")
        st.code(str(exc))
        st.stop()

    if diagnostics.empty:
        st.warning("The model diagnostics dataset is empty.")
        st.stop()
    d = diagnostics.iloc[0]
    st.subheader("Model overview")
    st.write(f"**Model:** {d['model_family']} | **Link:** {d['link']} | **Outcome:** {d['outcome']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input cohort patients", f"{int(d['input_cohort_patients']):,}")
    c2.metric("Adults in model with complete windows", f"{int(d['adult_complete_window_model_patients']):,}")
    c3.metric("Mean post-index encounters", f"{float(d['mean_post_encounters']):.2f}")
    c4.metric("Pearson dispersion", f"{float(d['pearson_dispersion']):.2f}")
    if bool(d["overdispersion_flag"]):
        st.warning(
            "Overdispersion was flagged. Poisson standard errors, confidence intervals "
            "and exploratory p-values may not be reliable; interpret inferential results cautiously."
        )

    st.subheader("Model effects: incidence rate ratios")
    st.caption(
        "An incidence rate ratio (IRR) above 1 indicates a higher modeled encounter rate; "
        "below 1 indicates a lower rate, conditional on the other model terms. "
        "Intervals are displayed as stored in the AWS model outputs."
    )
    coefficient_fields = [
        "term", "incidence_rate_ratio", "irr_ci95_lower", "irr_ci95_upper",
        "coefficient_log_rate", "standard_error", "p_value_exploratory",
        "uncertainty_method", "reference_group",
    ]
    coeff = coefficients[coefficient_fields].copy()
    for col in (
        "incidence_rate_ratio", "irr_ci95_lower", "irr_ci95_upper",
        "coefficient_log_rate", "standard_error", "p_value_exploratory",
    ):
        coeff[col] = pd.to_numeric(coeff[col], errors="coerce")
    plotted = coeff.dropna(subset=["term", "incidence_rate_ratio", "irr_ci95_lower", "irr_ci95_upper"]).copy()
    plotted = plotted[
        (plotted["incidence_rate_ratio"] > 0)
        & (plotted["irr_ci95_lower"] > 0)
        & (plotted["irr_ci95_upper"] >= plotted["incidence_rate_ratio"])
        & (plotted["incidence_rate_ratio"] >= plotted["irr_ci95_lower"])
    ]
    if not plotted.empty:
        import plotly.graph_objects as go
        plotted = plotted.sort_values("incidence_rate_ratio")
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=plotted["incidence_rate_ratio"],
            y=plotted["term"].astype(str),
            mode="markers",
            marker={"size": 11},
            error_x={
                "type": "data", "symmetric": False,
                "array": (plotted["irr_ci95_upper"] - plotted["incidence_rate_ratio"]).tolist(),
                "arrayminus": (plotted["incidence_rate_ratio"] - plotted["irr_ci95_lower"]).tolist(),
            },
            name="IRR and reported 95% interval",
        ))
        figure.add_vline(x=1, line_dash="dash", line_color="gray")
        figure.update_layout(
            title="Modeled encounter rate ratios with reported 95% intervals",
            xaxis_title="Incidence rate ratio (IRR)",
            yaxis_title="Model term",
            height=max(350, 90 + len(plotted) * 48),
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("No plottable coefficient intervals are available.")
    st.dataframe(
        coeff.rename(columns={
            "term": "Model term", "incidence_rate_ratio": "IRR",
            "irr_ci95_lower": "95% CI lower", "irr_ci95_upper": "95% CI upper",
            "coefficient_log_rate": "Log-rate coefficient", "standard_error": "Standard error",
            "p_value_exploratory": "Exploratory p-value",
            "uncertainty_method": "Uncertainty method", "reference_group": "Reference group",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Model diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed zero fraction", f"{float(d['observed_zero_fraction']):.1%}")
    c2.metric("Poisson expected zero fraction", f"{float(d['poisson_expected_zero_fraction']):.1%}")
    c3.metric("Residual degrees of freedom", f"{int(d['residual_degrees_of_freedom']):,}")
    diag_view = pd.DataFrame({
        "Metric": ["Deviance", "Pearson chi-square", "Pearson dispersion"],
        "Value": [float(d["deviance"]), float(d["pearson_chi_square"]), float(d["pearson_dispersion"])],
    })
    st.dataframe(diag_view, hide_index=True, use_container_width=True)
    st.caption("Stored modeling limitations: " + str(d["interpretation_limitations"]))

    st.subheader("Utilization and costs by number of conditions")
    segment_metrics = {
        "Mean encounters per patient": ("mean_pre_encounters_180d", "mean_post_encounters_180d"),
        "Mean encounter claim cost ($)": ("mean_pre_claim_cost_180d", "mean_post_claim_cost_180d"),
    }
    metric = st.selectbox("Compare model segments", list(segment_metrics), key="model_segment_metric")
    pre, post = segment_metrics[metric]
    segment_chart = segments[["disease_count", "patient_count", pre, post]].copy()
    for column in [pre, post]:
        segment_chart[column] = pd.to_numeric(segment_chart[column], errors="coerce")
    segment_chart["Conditions"] = segment_chart["disease_count"].astype(str)
    segment_long = segment_chart.melt(
        id_vars=["Conditions", "patient_count"], value_vars=[pre, post],
        var_name="Period", value_name="Average",
    )
    segment_long["Period"] = segment_long["Period"].map({
        pre: "180 days before", post: "180 days after",
    })
    fig_segment = px.bar(
        segment_long, x="Conditions", y="Average", color="Period", barmode="group",
        title=metric + " by number of chronic conditions",
        labels={"Conditions": "Number of conditions"},
    )
    st.plotly_chart(fig_segment, use_container_width=True)
    st.dataframe(
        segment_chart.rename(columns={
            "disease_count": "Number of conditions", "patient_count": "Patients",
            pre: "Mean before", post: "Mean after",
        })[["Number of conditions", "Patients", "Mean before", "Mean after"]],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Results are based entirely on synthetic records. Comparisons are observational; "
        "the model does not establish causation or demonstrate real-world clinical validity."
    )
else:
    st.info("This dashboard section is under development. Curated AWS data will be connected next.")
