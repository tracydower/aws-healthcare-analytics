"""Build and validate the Synthea encounter-level longitudinal model in AWS Glue.

Inputs: s3://healthcare-analytics-td-2026/processed/{patients,encounters,claims,conditions,procedures}/
Output: s3://healthcare-analytics-td-2026/curated/encounter_longitudinal/

One output row per (source_dataset, encounter id). Every source count must reconcile
before the job overwrites the existing curated dataset.
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, functions as F
from awsglue.context import GlueContext

# Initialize Glue's Spark runtime.
sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

# Keep the bucket overridable without requiring additional Glue job arguments.
args = getResolvedOptions(sys.argv, ["JOB_NAME"]) if "--JOB_NAME" in sys.argv else {}
BUCKET = "healthcare-analytics-td-2026"
BASE = f"s3://{BUCKET}"


def read_processed(table):
    """Read one previously standardized Parquet dataset."""
    return spark.read.parquet(f"{BASE}/processed/{table}/")


def assert_no_bad_keys(df, name, key):
    """Ensure a parent table has nonblank unique source-qualified keys."""
    invalid = df.filter(
        F.col("source_dataset").isNull()
        | F.col(key).isNull()
        | (F.trim(F.col(key)) == "")
    ).count()
    duplicates = (
        df.groupBy("source_dataset", key)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if invalid or duplicates:
        raise ValueError(f"{name}: invalid keys={invalid}, duplicate keys={duplicates}")


def assert_relationship(child, child_key, parent, parent_key, label):
    """Reject orphan child rows before performing any aggregations or joins."""
    parents = parent.select(
        "source_dataset", F.col(parent_key).alias("_parent_key")
    ).distinct()
    orphans = child.join(
        parents,
        (child["source_dataset"] == parents["source_dataset"])
        & (child[child_key] == parents["_parent_key"]),
        "left_anti",
    ).count()
    if orphans:
        raise ValueError(f"{label}: {orphans} unmatched child records")


def check_equal(actual, expected, label):
    """Fail the job instead of publishing an unreconciled model."""
    if actual != expected:
        raise ValueError(f"{label}: expected {expected:,}; found {actual:,}")
    print(f"PASS {label}: {actual:,}")


# Read five source datasets, preserving source_dataset as part of every key.
patients = read_processed("patients")
encounters = read_processed("encounters")
claims = read_processed("claims")
conditions = read_processed("conditions")
procedures = read_processed("procedures")

# Validate primary keys and each source-qualified foreign-key relationship.
assert_no_bad_keys(patients, "patients", "Id")
assert_no_bad_keys(encounters, "encounters", "id")
assert_no_bad_keys(claims, "claims", "id")
assert_relationship(encounters, "patient", patients, "Id", "encounters -> patients")
assert_relationship(claims, "patientid", patients, "Id", "claims -> patients")
assert_relationship(claims, "appointmentid", encounters, "id", "claims -> encounters")
assert_relationship(conditions, "encounter", encounters, "id", "conditions -> encounters")
assert_relationship(procedures, "encounter", encounters, "id", "procedures -> encounters")

# Verify that a claim and its matched encounter belong to the same patient.
claim_encounter = claims.alias("c").join(
    encounters.alias("e"),
    (F.col("c.source_dataset") == F.col("e.source_dataset"))
    & (F.col("c.appointmentid") == F.col("e.id")),
    "inner",
)
patient_mismatches = claim_encounter.filter(
    F.col("c.patientid") != F.col("e.patient")
).count()
check_equal(patient_mismatches, 0, "claim-to-encounter patient mismatches")

# Aggregate all one-to-many child tables BEFORE joining to encounter grain.
claim_summary = claims.groupBy("source_dataset", "appointmentid").agg(
    F.count("*").alias("claim_count")
).withColumnRenamed("appointmentid", "id")
condition_summary = conditions.groupBy("source_dataset", "encounter").agg(
    F.count("*").alias("condition_count"),
    F.countDistinct("code").alias("distinct_condition_codes"),
).withColumnRenamed("encounter", "id")
procedure_summary = procedures.groupBy("source_dataset", "encounter").agg(
    F.count("*").alias("procedure_count"),
    F.countDistinct("code").alias("distinct_procedure_codes"),
).withColumnRenamed("encounter", "id")

# Select only verified demographic attributes; do not infer survival status.
patient_dim = patients.select(
    "source_dataset",
    F.col("Id").alias("patient_id"),
    F.col("BIRTHDATE").alias("birthdate"),
    F.col("DEATHDATE").alias("deathdate"),
    F.col("GENDER").alias("gender"),
    F.col("RACE").alias("race"),
    F.col("ETHNICITY").alias("ethnicity"),
)

# Join against renamed keys to avoid ambiguous or duplicate column names.
model = encounters.join(
    patient_dim,
    (encounters["source_dataset"] == patient_dim["source_dataset"])
    & (encounters["patient"] == patient_dim["patient_id"]),
    "inner",
).drop(patient_dim["source_dataset"])

for summary in (claim_summary, condition_summary, procedure_summary):
    model = model.join(summary, on=["source_dataset", "id"], how="left")

# Convert the fields needed by downstream analytical models.
model = (
    model.withColumn("encounter_date", F.to_date(F.substring("start", 1, 10)))
    .withColumn("encounter_end_date", F.to_date(F.substring("stop", 1, 10)))
    .withColumn("total_claim_cost", F.expr("try_cast(total_claim_cost as double)"))
    .withColumn("base_encounter_cost", F.expr("try_cast(base_encounter_cost as double)"))
    .withColumn("payer_coverage", F.expr("try_cast(payer_coverage as double)"))
    .withColumn("claim_count", F.coalesce("claim_count", F.lit(0)))
    .withColumn("condition_count", F.coalesce("condition_count", F.lit(0)))
    .withColumn("procedure_count", F.coalesce("procedure_count", F.lit(0)))
    .withColumn("distinct_condition_codes", F.coalesce("distinct_condition_codes", F.lit(0)))
    .withColumn("distinct_procedure_codes", F.coalesce("distinct_procedure_codes", F.lit(0)))
)

# Validate all counts and keys BEFORE overwriting existing curated output.
expected_encounters = encounters.count()
expected_claims = claims.count()
expected_conditions = conditions.count()
expected_procedures = procedures.count()
summary = model.agg(
    F.count("*").alias("encounters"),
    F.sum("claim_count").alias("claims"),
    F.sum("condition_count").alias("conditions"),
    F.sum("procedure_count").alias("procedures"),
    F.sum(F.col("encounter_date").isNull().cast("int")).alias("missing_encounter_dates"),
).first()
check_equal(summary["encounters"], expected_encounters, "encounter row count")
check_equal(summary["claims"], expected_claims, "claim count")
check_equal(summary["conditions"], expected_conditions, "condition count")
check_equal(summary["procedures"], expected_procedures, "procedure count")
check_equal(summary["missing_encounter_dates"], 0, "missing encounter dates")
check_equal(
    model.select("source_dataset", "id").distinct().count(),
    expected_encounters,
    "unique source-qualified encounter IDs",
)

# Publish only after every reconciliation succeeds.
output = f"{BASE}/curated/encounter_longitudinal/"
model.write.mode("overwrite").parquet(output)

# Verify the persisted result, not just the in-memory DataFrame.
saved = spark.read.parquet(output)
check_equal(saved.count(), expected_encounters, "persisted encounter count")
print(f"SUCCESS: validated model published to {output}")
