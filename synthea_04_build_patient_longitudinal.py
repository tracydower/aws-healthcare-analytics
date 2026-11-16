"""Build a validated, patient-grain longitudinal analytics dataset with AWS Glue 6.0.

Inputs:
  s3://healthcare-analytics-td-2026/processed/patients/
  s3://healthcare-analytics-td-2026/curated/encounter_longitudinal/
Output:
  s3://healthcare-analytics-td-2026/curated/patient_longitudinal/

Each output row represents exactly one (source_dataset, patient ID).
Patients with no encounters are retained. Missing death dates mean unknown vital
status, not confirmed survival. Every metric is reconciled before publication.
"""

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from awsglue.context import GlueContext

# Initialize the AWS Glue Spark session.
sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

# Define stable source and target locations.
BUCKET = "healthcare-analytics-td-2026"
ROOT = f"s3://{BUCKET}"
PATIENTS_PATH = f"{ROOT}/processed/patients/"
ENCOUNTERS_PATH = f"{ROOT}/curated/encounter_longitudinal/"
OUTPUT_PATH = f"{ROOT}/curated/patient_longitudinal/"


def check_equal(actual, expected, label):
    # Raise an error before publication if a required reconciliation fails.
    if actual != expected:
        raise ValueError(f"FAIL {label}: expected={expected:,}, actual={actual:,}")
    print(f"PASS {label}: {actual:,}")


def check_keys(df, name, key):
    # Check source-qualified identifiers for missing values or duplicates.
    bad = df.filter(
        F.col("source_dataset").isNull()
        | F.col(key).isNull()
        | (F.trim(F.col(key)) == "")
    ).count()
    duplicate_groups = df.groupBy("source_dataset", key).count().filter("count > 1").count()
    check_equal(bad, 0, f"{name} invalid keys")
    check_equal(duplicate_groups, 0, f"{name} duplicate key groups")


# Load processed patients and the previously validated encounter-level model.
patients = spark.read.parquet(PATIENTS_PATH)
encounters = spark.read.parquet(ENCOUNTERS_PATH)

# Assert that both source tables have one record per source-qualified ID.
check_keys(patients, "patients", "Id")
check_keys(encounters, "encounters", "id")

# Verify that each encounter links to an existing patient in its own source.
patient_keys = patients.select(
    "source_dataset", F.col("Id").alias("patient_key")
)
orphans = encounters.alias("e").join(
    patient_keys.alias("p"),
    (F.col("e.source_dataset") == F.col("p.source_dataset"))
    & (F.col("e.patient") == F.col("p.patient_key")),
    "left_anti",
).count()
check_equal(orphans, 0, "unmatched encounter-to-patient keys")

# Ensure the curated encounter model has valid dates and expected count fields.
missing_dates = encounters.filter(F.col("encounter_date").isNull()).count()
check_equal(missing_dates, 0, "encounters with missing encounter dates")
metric_columns = ["claim_count", "condition_count", "procedure_count"]
negative_or_missing = encounters.filter(
    F.col("claim_count").isNull()
    | F.col("condition_count").isNull()
    | F.col("procedure_count").isNull()
    | (F.col("claim_count") < 0)
    | (F.col("condition_count") < 0)
    | (F.col("procedure_count") < 0)
).count()
check_equal(negative_or_missing, 0, "invalid encounter record-count metrics")

# Derive the observation cutoff from the data, not from today's calendar date.
reference_date = encounters.agg(F.max("encounter_date").alias("max_date")).first()[0]
if reference_date is None:
    raise ValueError("Encounter input has no observation cutoff date")
print(f"Study observation cutoff: {reference_date}")

# Aggregate encounter-grain records to the patient's source-qualified key.
patient_summary = encounters.groupBy("source_dataset", "patient").agg(
    F.count("*").alias("encounter_count"),
    F.min("encounter_date").alias("first_encounter_date"),
    F.max("encounter_date").alias("last_encounter_date"),
    F.countDistinct(F.year("encounter_date")).alias("years_with_encounters"),
    F.sum(F.when(F.lower(F.col("encounterclass")) == "emergency", 1).otherwise(0)).alias("emergency_encounters"),
    F.sum(F.when(F.lower(F.col("encounterclass")) == "inpatient", 1).otherwise(0)).alias("inpatient_encounters"),
    F.sum(F.when(F.lower(F.col("encounterclass")) == "outpatient", 1).otherwise(0)).alias("outpatient_encounters"),
    F.sum("claim_count").alias("claim_count"),
    F.sum("condition_count").alias("condition_count"),
    F.sum("procedure_count").alias("procedure_count"),
    F.sum(F.coalesce("total_claim_cost", F.lit(0.0))).alias("total_encounter_claim_cost"),
    F.sum(F.coalesce("payer_coverage", F.lit(0.0))).alias("total_payer_coverage"),
)

# Preserve all original patient columns (including source_dataset) and retain
# patients who have no encounters, without introducing duplicate join keys.
summary_renamed = patient_summary.withColumnRenamed("patient", "patient_summary_id")
model = patients.alias("p").join(
    summary_renamed.alias("s"),
    (F.col("p.source_dataset") == F.col("s.source_dataset"))
    & (F.col("p.Id") == F.col("s.patient_summary_id")),
    "left",
).drop(F.col("s.source_dataset")).drop("patient_summary_id")

# Zero encounter/record counts mean the patient had no such observed records.
count_fields = [
    "encounter_count", "years_with_encounters", "emergency_encounters",
    "inpatient_encounters", "outpatient_encounters", "claim_count",
    "condition_count", "procedure_count",
]
for col_name in count_fields:
    model = model.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)))

# Add observation-based measures; missing death dates remain unknown.
model = (
    model.withColumn("observation_cutoff_date", F.lit(reference_date).cast("date"))
    .withColumn(
        "days_since_last_encounter",
        F.datediff(F.col("observation_cutoff_date"), F.col("last_encounter_date")),
    )
    .withColumn(
        "death_date_recorded",
        F.col("DEATHDATE").isNotNull(),
    )
)

# Reconcile patient-grain counts and every encounter-derived volume metric.
expected_patients = patients.count()
expected_encounters = encounters.count()
expected = encounters.agg(
    F.sum("claim_count").alias("claims"),
    F.sum("condition_count").alias("conditions"),
    F.sum("procedure_count").alias("procedures"),
).first()
actual = model.agg(
    F.count("*").alias("patients"),
    F.sum("encounter_count").alias("encounters"),
    F.sum("claim_count").alias("claims"),
    F.sum("condition_count").alias("conditions"),
    F.sum("procedure_count").alias("procedures"),
).first()
check_equal(actual["patients"], expected_patients, "patient row count")
check_equal(actual["encounters"], expected_encounters, "total encounters")
for name in ("claims", "conditions", "procedures"):
    check_equal(actual[name], expected[name], f"total {name}")
check_keys(model, "patient model", "Id")

# Validate patient counts within each original source population.
source_input_counts = {
    row["source_dataset"]: row["count"]
    for row in patients.groupBy("source_dataset").count().collect()
}
source_output_counts = {
    row["source_dataset"]: row["count"]
    for row in model.groupBy("source_dataset").count().collect()
}
if source_input_counts != source_output_counts:
    raise ValueError(f"Source reconciliation failed: {source_input_counts} != {source_output_counts}")
print(f"PASS source dataset populations: {source_output_counts}")

# Publish the validated patient-grain result to curated S3 Parquet.
model.write.mode("overwrite").parquet(OUTPUT_PATH)

# Verify persisted counts independently of the in-memory DataFrame.
saved = spark.read.parquet(OUTPUT_PATH)
check_equal(saved.count(), expected_patients, "persisted patient rows")
check_equal(saved.select("source_dataset", "Id").distinct().count(), expected_patients, "persisted unique patients")
print(f"SUCCESS: patient-level longitudinal model saved to {OUTPUT_PATH}")
