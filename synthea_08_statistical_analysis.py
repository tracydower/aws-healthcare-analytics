"""AWS Glue 6.0 / PySpark: descriptive, adjusted chronic-disease utilization analysis.

Input:
  s3://healthcare-analytics-td-2026/curated/chronic_disease_cohorts/

Outputs (Parquet):
  s3://healthcare-analytics-td-2026/curated/statistical_analysis/model_coefficients/
  s3://healthcare-analytics-td-2026/curated/statistical_analysis/model_diagnostics/
  s3://healthcare-analytics-td-2026/curated/statistical_analysis/segment_summary/

The cohort uses the earliest *recorded* diagnosis of three selected conditions.
These are synthetic observational records; a dataset-wide calendar window does
not establish continuous patient observation. Disease burden is computed from
conditions ever recorded, including diagnoses after the index date, so this is
an association analysis, NOT an at-index prediction or a causal assessment.
"""

import json
import math
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession, functions as F, types as T

# Initialize Spark and keep informational Spark messages out of job output.
spark = SparkSession.builder.appName("synthea_08_statistical_analysis").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Use project-specific, reproducible input and output paths.
ROOT = "s3://healthcare-analytics-td-2026/curated/"
INPUT_PATH = ROOT + "chronic_disease_cohorts/"
OUTPUT_ROOT = ROOT + "statistical_analysis/"
COEFFICIENT_PATH = OUTPUT_ROOT + "model_coefficients/"
DIAGNOSTIC_PATH = OUTPUT_ROOT + "model_diagnostics/"
SEGMENT_PATH = OUTPUT_ROOT + "segment_summary/"

# Require at least this many subjects in each modeled disease-burden group.
MIN_PATIENTS_PER_GROUP = 20
# Deviance alone is not a dispersion estimate; use Pearson chi-square / df.
OVERDISPERSION_FLAG = 1.5

# Read the curated patient-level chronic disease model built in job 07.
source = spark.read.parquet(INPUT_PATH)

# Explicitly check the exact source columns required for statistical modeling.
required_columns = [
    "source_dataset", "patient_id", "complete_both_calendar_windows",
    "age_at_index", "gender", "disease_count", "pre_encounters_180d",
    "post_encounters_180d", "cohort_segment",
]
missing_columns = [c for c in required_columns if c not in source.columns]
if missing_columns:
    raise ValueError("Missing expected columns: " + ", ".join(missing_columns))

# Verify one row per source-qualified patient before any statistical calculations.
if source.filter(F.col("patient_id").isNull() | F.col("source_dataset").isNull()).limit(1).count():
    raise ValueError("Missing source-qualified patient keys")
if source.groupBy("source_dataset", "patient_id").count().filter("count > 1").limit(1).count():
    raise ValueError("Duplicate patient keys in curated disease cohort input")

# Retain adults with both calendar windows available and recognized gender values.
# Missing data are excluded from the model explicitly, not silently imputed.
eligible = source.filter(
    (F.col("complete_both_calendar_windows") == 1)
    & (F.col("age_at_index") >= 18)
    & F.upper(F.trim("gender")).isin("F", "M")
    & F.col("disease_count").isin(1, 2, 3)
    & F.col("age_at_index").isNotNull()
    & F.col("pre_encounters_180d").isNotNull()
    & F.col("post_encounters_180d").isNotNull()
    & (F.col("pre_encounters_180d") >= 0)
    & (F.col("post_encounters_180d") >= 0)
)

# Use two indicator variables instead of assuming a linear burden effect.
# One chronic condition is the reference category for both comparisons.
model_rows = eligible.select(
    "source_dataset", "patient_id", "cohort_segment", "disease_count",
    F.col("post_encounters_180d").cast("double").alias("label"),
    F.when(F.col("disease_count") == 2, 1.0).otherwise(0.0).alias("two_conditions"),
    F.when(F.col("disease_count") == 3, 1.0).otherwise(0.0).alias("three_conditions"),
    F.col("age_at_index").cast("double").alias("age_at_index"),
    F.col("pre_encounters_180d").cast("double").alias("prior_encounters"),
    F.when(F.upper(F.trim("gender")) == "F", 1.0).otherwise(0.0).alias("female"),
)
model_rows = model_rows.cache()

# Check all groups have enough patients to support the planned comparison.
group_counts = {int(r["disease_count"]): int(r["count"]) for r in model_rows.groupBy("disease_count").count().collect()}
for group in (1, 2, 3):
    if group_counts.get(group, 0) < MIN_PATIENTS_PER_GROUP:
        raise ValueError("Insufficient adults with complete windows in burden group {}: {}".format(group, group_counts.get(group, 0)))

# Prepare a Spark ML vector in a documented, stable coefficient order.
feature_names = ["two_conditions", "three_conditions", "age_at_index", "prior_encounters", "female"]
assembler = VectorAssembler(inputCols=feature_names, outputCol="features", handleInvalid="error")
training = assembler.transform(model_rows).select("label", "features")

# Fit an adjusted log-linked Poisson generalized linear model.
# This is descriptive association modeling, with no train/test prediction claim.
model = GeneralizedLinearRegression(
    family="poisson", link="log", featuresCol="features", labelCol="label",
    fitIntercept=True, maxIter=100, tol=1e-7,
).fit(training)

# Generate patient-level fitted means for diagnostics, without publishing IDs.
fitted = model.transform(training).select(
    F.col("label"), F.col("prediction").alias("fitted_mean")
).cache()

# Calculate Pearson dispersion, zero-event calibration, and mean outcome.
# The model's default Poisson dispersion of 1 is an assumption, not a diagnostic.
stats = fitted.agg(
    F.count("*").alias("n"),
    F.avg("label").alias("mean_post_encounters"),
    F.avg(F.when(F.col("label") == 0, 1.0).otherwise(0.0)).alias("observed_zero_fraction"),
    F.avg(F.exp(-F.col("fitted_mean"))).alias("poisson_expected_zero_fraction"),
    F.sum(
        F.pow(F.col("label") - F.col("fitted_mean"), 2) /
        F.greatest(F.col("fitted_mean"), F.lit(1e-12))
    ).alias("pearson_chi_square"),
).first()

# Account for five covariates plus the intercept when computing residual df.
n = int(stats["n"])
residual_df = n - (len(feature_names) + 1)
if residual_df <= 0:
    raise ValueError("Not enough observations for the fitted model")
pearson_chi_square = float(stats["pearson_chi_square"])
dispersion = pearson_chi_square / residual_df
if not math.isfinite(dispersion) or dispersion <= 0:
    raise ValueError("Invalid Poisson dispersion statistic")

# Calculate quasi-Poisson scale-adjusted standard errors as a sensitivity
# analysis. These are NOT heteroskedasticity-robust or negative-binomial errors.
# Scale only upward; do not reduce estimated uncertainty for underdispersion.
variance_scale = max(1.0, dispersion)
standard_errors = [float(v) for v in model.summary.coefficientStandardErrors]
coefficients = [float(model.intercept)] + [float(v) for v in model.coefficients]
parameter_names = ["intercept"] + feature_names
if len(standard_errors) != len(coefficients):
    raise ValueError("Standard error and coefficient counts do not align")

# Standard-normal two-sided p-values can be calculated without extra packages.
def normal_two_sided_p(z_value):
    return math.erfc(abs(z_value) / math.sqrt(2.0))

# Create one interpretation-ready record per coefficient, including IRRs.
coefficient_records = []
for name, beta, se in zip(parameter_names, coefficients, standard_errors):
    if not math.isfinite(beta) or not math.isfinite(se) or se <= 0:
        raise ValueError("Nonfinite or invalid model coefficient: " + name)
    adjusted_se = se * math.sqrt(variance_scale)
    z_value = beta / adjusted_se
    lower_beta = beta - 1.96 * adjusted_se
    upper_beta = beta + 1.96 * adjusted_se
    coefficient_records.append((
        name, beta, math.exp(beta), adjusted_se,
        math.exp(lower_beta), math.exp(upper_beta),
        normal_two_sided_p(z_value),
        "quasi_poisson_scaled" if dispersion > 1.0 else "poisson_model_based",
        "1 condition" if name in ("two_conditions", "three_conditions") else
        ("male" if name == "female" else None),
    ))

# Persist a transparent fixed-schema coefficients table for BI/reporting.
coefficient_schema = T.StructType([
    T.StructField("term", T.StringType(), False),
    T.StructField("coefficient_log_rate", T.DoubleType(), False),
    T.StructField("incidence_rate_ratio", T.DoubleType(), False),
    T.StructField("standard_error", T.DoubleType(), False),
    T.StructField("irr_ci95_lower", T.DoubleType(), False),
    T.StructField("irr_ci95_upper", T.DoubleType(), False),
    T.StructField("p_value_exploratory", T.DoubleType(), False),
    T.StructField("uncertainty_method", T.StringType(), False),
    T.StructField("reference_group", T.StringType(), True),
])
coefficients_df = spark.createDataFrame(coefficient_records, coefficient_schema)

# Record source population, model sample, diagnostics and scientific caveats.
diagnostic_data = [(
    int(source.count()), n, group_counts[1], group_counts[2], group_counts[3],
    "poisson", "log", "post_encounters_180d", int(residual_df),
    float(model.summary.deviance), pearson_chi_square, dispersion,
    bool(dispersion > OVERDISPERSION_FLAG),
    float(stats["mean_post_encounters"]),
    float(stats["observed_zero_fraction"]),
    float(stats["poisson_expected_zero_fraction"]),
    "Synthetic observational data; associations only. First recorded diagnosis != onset. "
    "Calendar bounds != continuous enrollment. Disease count includes future diagnoses; "
    "do not use this model as an at-index prediction. Quasi-Poisson-scaled "
    "uncertainty is exploratory and does not replace robust or NB inference.",
)]
diagnostics_columns = [
    "input_cohort_patients", "adult_complete_window_model_patients",
    "patients_one_condition", "patients_two_conditions", "patients_three_conditions",
    "model_family", "link", "outcome", "residual_degrees_of_freedom",
    "deviance", "pearson_chi_square", "pearson_dispersion", "overdispersion_flag",
    "mean_post_encounters", "observed_zero_fraction", "poisson_expected_zero_fraction",
    "interpretation_limitations",
]
diagnostics_df = spark.createDataFrame(diagnostic_data, diagnostics_columns)

# Report sample sizes and raw pre/post encounter averages by burden group.
segment_df = eligible.groupBy("disease_count").agg(
    F.count("*").alias("patient_count"),
    F.avg("age_at_index").alias("mean_age_at_index"),
    F.avg("pre_encounters_180d").alias("mean_pre_encounters_180d"),
    F.avg("post_encounters_180d").alias("mean_post_encounters_180d"),
    F.avg("pre_claim_cost_180d").alias("mean_pre_claim_cost_180d"),
    F.avg("post_claim_cost_180d").alias("mean_post_claim_cost_180d"),
)

# Reconcile output sample sizes before replacing the three result tables.
if segment_df.agg(F.sum("patient_count")).first()[0] != n:
    raise ValueError("Segment sample sizes do not reconcile with model sample")
if coefficients_df.count() != len(feature_names) + 1:
    raise ValueError("Unexpected number of regression coefficients")

# Save BI-ready model artifacts in their own repeatable Parquet locations.
coefficients_df.write.mode("overwrite").parquet(COEFFICIENT_PATH)
diagnostics_df.write.mode("overwrite").parquet(DIAGNOSTIC_PATH)
segment_df.write.mode("overwrite").parquet(SEGMENT_PATH)

# Verify successful output readback before reporting completion.
if spark.read.parquet(COEFFICIENT_PATH).count() != 6:
    raise ValueError("Coefficient output readback failed")
if spark.read.parquet(DIAGNOSTIC_PATH).count() != 1:
    raise ValueError("Diagnostic output readback failed")
if spark.read.parquet(SEGMENT_PATH).agg(F.sum("patient_count")).first()[0] != n:
    raise ValueError("Segment output readback failed")

# Emit a compact machine-readable operational summary to AWS Glue output logs.
print("STATISTICAL_ANALYSIS_RESULT=" + json.dumps({
    "model_patients": n,
    "disease_count_groups": group_counts,
    "pearson_dispersion": round(dispersion, 4),
    "overdispersion_flag": dispersion > OVERDISPERSION_FLAG,
    "coefficients_path": COEFFICIENT_PATH,
    "diagnostics_path": DIAGNOSTIC_PATH,
    "segment_summary_path": SEGMENT_PATH,
    "interpretation": "Synthetic observational associations, not causal or predictive validation",
}, sort_keys=True))

# Release cached data and shut down the Spark session.
fitted.unpersist()
model_rows.unpersist()
spark.stop()
