"""AWS Glue 6.0 / PySpark: overlapping chronic disease cohorts and descriptive patient outcomes.

Sources:
  s3://healthcare-analytics-td-2026/processed/patients/
  s3://healthcare-analytics-td-2026/processed/conditions/
  s3://healthcare-analytics-td-2026/processed/encounters/

Outputs:
  s3://healthcare-analytics-td-2026/curated/chronic_disease_cohorts/
  s3://healthcare-analytics-td-2026/curated/chronic_disease_cohort_summary/

First-recorded diagnosis is not necessarily disease onset. Pre/post comparisons are
observational and noncausal. Calendar coverage is a dataset-level proxy, NOT proof
of patient-level enrollment or continuous observation.
"""

import json
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("synthea_07_chronic_disease_cohorts").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

ROOT = "s3://healthcare-analytics-td-2026/"
PATIENT_PATH = ROOT + "processed/patients/"
CONDITION_PATH = ROOT + "processed/conditions/"
ENCOUNTER_PATH = ROOT + "processed/encounters/"
COHORT_PATH = ROOT + "curated/chronic_disease_cohorts/"
SUMMARY_PATH = ROOT + "curated/chronic_disease_cohort_summary/"

# Define the selected clinical concepts. Hypertension must use its diagnosis
# code because the data has both "Hypertension" and "Essential hypertension
# (disorder)" for code 59621000. Other labels remain description-based.
# These are project-specific groupings, not comprehensive clinical code sets.
DIAGNOSIS_DESCRIPTIONS = {
    "obesity": "Body mass index 30+ - obesity (finding)",
    "prediabetes": "Prediabetes",
    "hypertension": "59621000",  # Diagnosis code, not a description.
}
WINDOW_DAYS = 180

# Read the same source-qualified datasets used by the earlier Glue ETL jobs.
patients = spark.read.parquet(PATIENT_PATH).select(
    F.col("source_dataset"), F.col("Id").alias("patient_id"),
    F.col("BIRTHDATE").cast("date").alias("birthdate"),
    F.col("DEATHDATE").cast("date").alias("deathdate"),
    F.col("GENDER").alias("gender"), F.col("RACE").alias("race"),
    F.col("ETHNICITY").alias("ethnicity"),
)
conditions = spark.read.parquet(CONDITION_PATH).select(
    F.col("source_dataset"), F.col("patient").alias("patient_id"),
    F.to_date("start").alias("diagnosis_date"),
    F.col("description").alias("diagnosis_description"),
    F.col("code").alias("diagnosis_code"),
)
encounters = spark.read.parquet(ENCOUNTER_PATH).select(
    F.col("source_dataset"), F.col("patient").alias("patient_id"),
    F.col("id").alias("encounter_id"),
    F.to_date("start").alias("encounter_date"),
    F.col("encounterclass").alias("encounter_class"),
    F.col("total_claim_cost").cast("double").alias("encounter_claim_cost"),
)

# Verify source grain before deriving the cohort; never join by patient_id alone.
assert patients.filter(F.col("patient_id").isNull() | F.col("source_dataset").isNull()).count() == 0, "Null patient key"
assert patients.groupBy("source_dataset", "patient_id").count().filter("count > 1").count() == 0, "Duplicate patient key"
assert encounters.groupBy("source_dataset", "encounter_id").count().filter("count > 1").count() == 0, "Duplicate encounter key"
assert encounters.filter(F.col("encounter_date").isNull()).count() == 0, "Unparseable encounter date"

# Label obesity and prediabetes by description, and hypertension by code.
condition_label = (
    F.when(F.col("diagnosis_description") == DIAGNOSIS_DESCRIPTIONS["obesity"], "obesity")
     .when(F.col("diagnosis_description") == DIAGNOSIS_DESCRIPTIONS["prediabetes"], "prediabetes")
     .when(F.trim(F.col("diagnosis_code")) == DIAGNOSIS_DESCRIPTIONS["hypertension"], "hypertension")
)
diagnoses = conditions.withColumn("disease", condition_label).filter(F.col("disease").isNotNull())
assert diagnoses.filter(F.col("diagnosis_date").isNull()).count() == 0, "Unparseable targeted diagnosis date"

# One first-recorded diagnosis per patient and disease, not one row per event.
first_diagnoses = diagnoses.groupBy("source_dataset", "patient_id").agg(
    *[
        F.min(F.when(F.col("disease") == disease, F.col("diagnosis_date"))).alias(disease + "_first_date")
        for disease in DIAGNOSIS_DESCRIPTIONS
    ]
)

# Use the earliest of the three diagnoses as a common index date per patient.
cohorts = patients.join(first_diagnoses, ["source_dataset", "patient_id"], "inner")
for disease in DIAGNOSIS_DESCRIPTIONS:
    cohorts = cohorts.withColumn("has_" + disease, F.col(disease + "_first_date").isNotNull().cast("int"))
cohorts = cohorts.withColumn(
    "disease_count", sum((F.col("has_" + disease) for disease in DIAGNOSIS_DESCRIPTIONS), F.lit(0))
).withColumn(
    "index_date", F.least(*[F.col(disease + "_first_date") for disease in DIAGNOSIS_DESCRIPTIONS])
).withColumn(
    "cohort_segment", F.concat_ws(" + ", *[
        F.when(F.col("has_" + disease) == 1, F.lit(disease.title())) for disease in DIAGNOSIS_DESCRIPTIONS
    ])
)

# Restrict cohort calendar windows to source dataset availability; this does NOT
# confirm that each patient had health coverage or was continuously observable.
calendar = encounters.groupBy("source_dataset").agg(
    F.min("encounter_date").alias("dataset_first_encounter_date"),
    F.max("encounter_date").alias("dataset_last_encounter_date"),
)
cohorts = cohorts.join(calendar, "source_dataset", "left").withColumn(
    "complete_pre_calendar_window",
    (F.date_sub(F.col("index_date"), WINDOW_DAYS) >= F.col("dataset_first_encounter_date")).cast("int"),
).withColumn(
    "complete_post_calendar_window",
    (F.date_add(F.col("index_date"), WINDOW_DAYS) <= F.col("dataset_last_encounter_date")).cast("int"),
).withColumn(
    "complete_both_calendar_windows",
    ((F.col("complete_pre_calendar_window") == 1) & (F.col("complete_post_calendar_window") == 1)).cast("int"),
)

# Keep one observation per encounter when aggregating utilization and costs.
# Inclusive/exclusive boundaries: pre = [index-180, index), post = (index, index+180].
# Index-date encounters are reported separately to avoid conflating diagnostic
# encounters with subsequent utilization.
joined = cohorts.select("source_dataset", "patient_id", "index_date").join(
    encounters, ["source_dataset", "patient_id"], "left"
)
days_since_index = F.datediff(F.col("encounter_date"), F.col("index_date"))
pre = (days_since_index >= -WINDOW_DAYS) & (days_since_index < 0)
post = (days_since_index > 0) & (days_since_index <= WINDOW_DAYS)
index_day = days_since_index == 0
nonnegative_cost = F.coalesce(F.col("encounter_claim_cost"), F.lit(0.0))
utilization = joined.groupBy("source_dataset", "patient_id").agg(
    F.sum(F.when(pre, 1).otherwise(0)).cast("long").alias("pre_encounters_180d"),
    F.sum(F.when(post, 1).otherwise(0)).cast("long").alias("post_encounters_180d"),
    F.sum(F.when(index_day, 1).otherwise(0)).cast("long").alias("index_day_encounters"),
    F.sum(F.when(pre, nonnegative_cost).otherwise(0.0)).alias("pre_claim_cost_180d"),
    F.sum(F.when(post, nonnegative_cost).otherwise(0.0)).alias("post_claim_cost_180d"),
    F.sum(F.when(index_day, nonnegative_cost).otherwise(0.0)).alias("index_day_claim_cost"),
    F.sum(F.when(pre & (F.lower("encounter_class") == "emergency"), 1).otherwise(0)).cast("long").alias("pre_emergency_encounters_180d"),
    F.sum(F.when(post & (F.lower("encounter_class") == "emergency"), 1).otherwise(0)).cast("long").alias("post_emergency_encounters_180d"),
    F.sum(F.when(pre & (F.lower("encounter_class") == "inpatient"), 1).otherwise(0)).cast("long").alias("pre_inpatient_encounters_180d"),
    F.sum(F.when(post & (F.lower("encounter_class") == "inpatient"), 1).otherwise(0)).cast("long").alias("post_inpatient_encounters_180d"),
)
result = cohorts.join(utilization, ["source_dataset", "patient_id"], "left")
result = result.withColumn("age_at_index", F.floor(F.months_between("index_date", "birthdate") / 12).cast("int"))

# Summary: mutually exclusive disease combinations, with complete-window
# subset averages reported separately from all-cohort descriptive averages.
complete = F.col("complete_both_calendar_windows") == 1
summary = result.groupBy("cohort_segment", "disease_count").agg(
    F.count("*").alias("patient_count"),
    F.sum("complete_both_calendar_windows").alias("patients_with_both_calendar_windows"),
    F.avg("pre_encounters_180d").alias("all_patients_avg_pre_encounters_180d"),
    F.avg("post_encounters_180d").alias("all_patients_avg_post_encounters_180d"),
    F.avg("pre_claim_cost_180d").alias("all_patients_avg_pre_claim_cost_180d"),
    F.avg("post_claim_cost_180d").alias("all_patients_avg_post_claim_cost_180d"),
    F.avg(F.when(complete, F.col("pre_encounters_180d"))).alias("complete_window_avg_pre_encounters_180d"),
    F.avg(F.when(complete, F.col("post_encounters_180d"))).alias("complete_window_avg_post_encounters_180d"),
    F.avg(F.when(complete, F.col("pre_claim_cost_180d"))).alias("complete_window_avg_pre_claim_cost_180d"),
    F.avg(F.when(complete, F.col("post_claim_cost_180d"))).alias("complete_window_avg_post_claim_cost_180d"),
)

# Validate the analytical grain and reference integrity before output writes.
expected_count = first_diagnoses.count()
assert result.count() == expected_count, "Cohort count changed during joins"
assert result.groupBy("source_dataset", "patient_id").count().filter("count > 1").count() == 0, "Duplicate cohort patient"
assert result.filter(F.col("index_date").isNull()).count() == 0, "Missing index date"
assert result.filter(~F.col("disease_count").between(1, 3)).count() == 0, "Invalid disease count"
assert summary.agg(F.sum("patient_count")).first()[0] == expected_count, "Summary patient counts do not reconcile"

# Save repeatable, independently queryable Parquet outputs.
result.write.mode("overwrite").parquet(COHORT_PATH)
summary.write.mode("overwrite").parquet(SUMMARY_PATH)
assert spark.read.parquet(COHORT_PATH).count() == expected_count, "Cohort output readback count mismatch"
assert spark.read.parquet(SUMMARY_PATH).agg(F.sum("patient_count")).first()[0] == expected_count, "Summary output readback mismatch"

print("CHRONIC_DISEASE_COHORT_RESULT=" + json.dumps({
    "patients": expected_count,
    "cohort_segments": summary.count(),
    "cohort_path": COHORT_PATH,
    "summary_path": SUMMARY_PATH,
    "window_days": WINDOW_DAYS,
    "window_caveat": "Calendar window within dataset range; does not establish patient enrollment/observation",
    "index_caveat": "First recorded diagnosis among selected conditions; not disease onset",
}, sort_keys=True))
spark.stop()
