"""AWS Glue 6.0: Build reproducible diagnosis-centered patient journey datasets.

Inputs:
  processed/conditions/, processed/procedures/, processed/patients/
  curated/encounter_longitudinal/
Outputs:
  curated/patient_disease_events/                one record per diagnosis/procedure event
  curated/patient_disease_journeys/              one row per patient and diagnosis code
  curated/disease_cohort_summary/                descriptive statistics per diagnosis code

Windows: PRE = index date minus 180 through minus 1 day; POST = index date
through plus 179 days. POST includes the index encounter and is NOT a causal
post-treatment comparison. The last encounter date in the combined dataset is
used as the observation cutoff; it does not prove continuous enrollment.
Synthea is synthetic; no clinical efficacy or causal claims are supported.
"""

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from awsglue.context import GlueContext

# Start the Spark session managed by AWS Glue.
sc = SparkContext.getOrCreate()
spark = GlueContext(sc).spark_session

# Set the project input and output locations.
ROOT = "s3://healthcare-analytics-td-2026"
CONDITIONS = f"{ROOT}/processed/conditions/"
PROCEDURES = f"{ROOT}/processed/procedures/"
PATIENTS = f"{ROOT}/processed/patients/"
ENCOUNTERS = f"{ROOT}/curated/encounter_longitudinal/"
EVENTS_OUTPUT = f"{ROOT}/curated/patient_disease_events/"
JOURNEYS_OUTPUT = f"{ROOT}/curated/patient_disease_journeys/"
COHORT_OUTPUT = f"{ROOT}/curated/disease_cohort_summary/"
WINDOW_DAYS = 180


def require_equal(actual, expected, description):
    # Fail immediately when an input or output fails reconciliation.
    if actual != expected:
        raise ValueError(f"FAIL {description}: actual={actual}, expected={expected}")
    print(f"PASS {description}: {actual}")


def check_unique(df, keys, description):
    # Reject duplicate natural keys before publishing results.
    duplicates = df.groupBy(*keys).count().filter(F.col("count") > 1).count()
    require_equal(duplicates, 0, f"{description} duplicate keys")


# Read the curated encounter grain and the original condition/procedure events.
e = spark.read.parquet(ENCOUNTERS)
c = spark.read.parquet(CONDITIONS)
p = spark.read.parquet(PROCEDURES)
patients = spark.read.parquet(PATIENTS)

# Verify parent table keys and required dates before joining event records.
check_unique(e, ["source_dataset", "id"], "encounters")
check_unique(patients, ["source_dataset", "Id"], "patients")
require_equal(e.filter(F.col("encounter_date").isNull()).count(), 0, "missing encounter dates")
cutoff = e.agg(F.max("encounter_date").alias("cutoff")).first()["cutoff"]
if cutoff is None:
    raise ValueError("No encounter dates are available for the observation cutoff")
print(f"Study observation cutoff: {cutoff}")

# Check condition/procedure rows link to a real source-qualified encounter.
parent_keys = e.select("source_dataset", F.col("id").alias("encounter_key"))
for source, name in [(c, "conditions"), (p, "procedures")]:
    orphans = source.alias("x").join(
        parent_keys.alias("y"),
        (F.col("x.source_dataset") == F.col("y.source_dataset"))
        & (F.col("x.encounter") == F.col("y.encounter_key")),
        "left_anti",
    ).count()
    require_equal(orphans, 0, f"orphan {name}")

# Attach verified encounter-level patient identity to each clinical event.
e_identity = e.select(
    "source_dataset", F.col("id").alias("encounter_key"),
    F.col("patient").alias("patient_id"), "encounter_date"
)

def to_events(source, event_type):
    # Keep event date from the clinical table; do not substitute encounter dates.
    mapped = source.alias("x").join(
        e_identity.alias("e"),
        (F.col("x.source_dataset") == F.col("e.source_dataset"))
        & (F.col("x.encounter") == F.col("e.encounter_key")),
        "inner",
    )
    return mapped.select(
        F.col("x.source_dataset").alias("source_dataset"),
        F.col("e.patient_id").alias("patient_id"),
        F.col("x.encounter").alias("encounter_id"),
        F.lit(event_type).alias("event_type"),
        F.to_date(F.substring(F.col("x.start"), 1, 10)).alias("event_date"),
        F.trim(F.col("x.code")).alias("event_code"),
        F.col("x.description").alias("event_description"),
    )

# Create the chronological event source for downstream patient journey views.
diagnoses = to_events(c, "DIAGNOSIS")
treatments = to_events(p, "PROCEDURE")
events = diagnoses.unionByName(treatments)
require_equal(events.count(), c.count() + p.count(), "all diagnosis/procedure event rows")
invalid_events = events.filter(
    F.col("event_date").isNull()
    | F.col("patient_id").isNull()
    | F.col("event_code").isNull()
    | (F.col("event_code") == "")
).count()
require_equal(invalid_events, 0, "invalid clinical event dates, patient IDs or codes")

# Find the first observed occurrence of each diagnosis code per patient.
index = diagnoses.groupBy("source_dataset", "patient_id", "event_code").agg(
    F.min("event_date").alias("index_diagnosis_date"),
    F.count("*").alias("diagnosis_record_count"),
    F.countDistinct("encounter_id").alias("diagnosis_encounter_count"),
)

# Pick a deterministic diagnosis description recorded on the index date.
description = diagnoses.groupBy("source_dataset", "patient_id", "event_code").agg(
    F.min(F.struct("event_date", "event_description")).alias("earliest")
).select(
    "source_dataset", "patient_id", "event_code",
    F.col("earliest.event_description").alias("diagnosis_description")
)
index = index.join(description, ["source_dataset", "patient_id", "event_code"])
check_unique(index, ["source_dataset", "patient_id", "event_code"], "diagnosis index")

# Derive each patient's first observed encounter as a history-length indicator.
first_seen = e.groupBy("source_dataset", "patient").agg(
    F.min("encounter_date").alias("first_observed_encounter_date")
).withColumnRenamed("patient", "patient_id")
index = index.join(first_seen, ["source_dataset", "patient_id"], "left")
index = index.withColumn("observation_cutoff_date", F.lit(cutoff).cast("date"))
index = index.withColumn(
    "has_180d_prior_recorded_history",
    F.col("first_observed_encounter_date") <= F.date_sub("index_diagnosis_date", WINDOW_DAYS),
).withColumn(
    "has_180d_calendar_followup",
    F.col("observation_cutoff_date") >= F.date_add("index_diagnosis_date", WINDOW_DAYS - 1),
)

# Attach each encounter to the diagnosis index for its own patient and source.
encounter_metrics = e.select(
    "source_dataset", F.col("patient").alias("patient_id"), "encounter_date",
    "encounterclass", "claim_count", "procedure_count", "condition_count",
    "total_claim_cost", "payer_coverage",
)
linked = index.select("source_dataset", "patient_id", "event_code", "index_diagnosis_date").alias("i").join(
    encounter_metrics.alias("e"),
    (F.col("i.source_dataset") == F.col("e.source_dataset"))
    & (F.col("i.patient_id") == F.col("e.patient_id"))
    & (F.col("e.encounter_date") >= F.date_sub(F.col("i.index_diagnosis_date"), WINDOW_DAYS))
    & (F.col("e.encounter_date") < F.date_add(F.col("i.index_diagnosis_date"), WINDOW_DAYS)),
    "left",
).select(
    F.col("i.source_dataset").alias("source_dataset"),
    F.col("i.patient_id").alias("patient_id"),
    F.col("i.event_code").alias("event_code"),
    F.col("i.index_diagnosis_date").alias("index_diagnosis_date"),
    F.col("e.encounter_date").alias("encounter_date"),
    F.col("e.encounterclass").alias("encounterclass"),
    F.col("e.claim_count").alias("claim_count"),
    F.col("e.procedure_count").alias("procedure_count"),
    F.col("e.total_claim_cost").alias("total_claim_cost"),
    F.col("e.payer_coverage").alias("payer_coverage"),
)

# Classify pre-index and post-index encounters into nonoverlapping windows.
linked = linked.withColumn(
    "period",
    F.when(F.col("encounter_date") < F.col("index_diagnosis_date"), "pre")
     .when(F.col("encounter_date").isNotNull(), "post"),
)

# Aggregate recorded utilization and costs without multiplying encounter rows.
metrics = linked.groupBy("source_dataset", "patient_id", "event_code").agg(
    *[
        F.sum(F.when(F.col("period") == period, F.lit(1)).otherwise(F.lit(0)))
        .alias(f"{period}_encounters_180d")
        for period in ("pre", "post")
    ],
    *[
        F.sum(F.when(
            (F.col("period") == period) & (F.lower("encounterclass") == cls),
            F.lit(1),
        ).otherwise(F.lit(0))).alias(f"{period}_{cls}_encounters_180d")
        for period in ("pre", "post") for cls in ("emergency", "inpatient", "outpatient")
    ],
    *[
        F.sum(F.when(F.col("period") == period, F.coalesce(F.col(original).cast("double"), F.lit(0.0)))
              .otherwise(F.lit(0.0))).alias(f"{period}_{output}_180d")
        for period in ("pre", "post")
        for original, output in (
            ("claim_count", "claims"), ("procedure_count", "procedures"),
            ("total_claim_cost", "encounter_claim_cost"), ("payer_coverage", "payer_coverage")
        )
    ],
)

# Combine the indexed diagnosis and the two observation windows.
journeys = index.join(metrics, ["source_dataset", "patient_id", "event_code"], "left")
check_unique(journeys, ["source_dataset", "patient_id", "event_code"], "patient diagnosis journeys")
require_equal(journeys.count(), index.count(), "one journey per patient/diagnosis")
require_equal(
    journeys.agg(F.sum("diagnosis_record_count").alias("count")).first()["count"],
    c.count(), "all diagnosis records reconciled",
)

# Summarize cohort patterns across patients sharing the same diagnosis code.
# These are DESCRIPTIVE utilization statistics, not treatment-effect estimates.
cohorts = journeys.groupBy("event_code").agg(
    F.min("diagnosis_description").alias("diagnosis_description"),
    F.count("*").alias("patient_diagnosis_instances"),
    F.countDistinct("source_dataset", "patient_id").alias("distinct_patients"),
    F.sum(F.col("has_180d_prior_recorded_history").cast("int")).alias("patients_with_prior_history"),
    F.sum(F.col("has_180d_calendar_followup").cast("int")).alias("patients_with_calendar_followup"),
    F.avg("pre_encounters_180d").alias("avg_pre_encounters_180d"),
    F.avg("post_encounters_180d").alias("avg_post_encounters_180d"),
    F.avg("pre_encounter_claim_cost_180d").alias("avg_pre_encounter_claim_cost_180d"),
    F.avg("post_encounter_claim_cost_180d").alias("avg_post_encounter_claim_cost_180d"),
)
require_equal(cohorts.agg(F.sum("patient_diagnosis_instances")).first()[0], index.count(), "cohort total reconciliation")

# Validate all three datasets BEFORE writing any curated outputs.
print(f"PASS: events={events.count():,}, journeys={journeys.count():,}, cohorts={cohorts.count():,}")

# Publish self-describing Parquet tables for analytical reporting.
events.write.mode("overwrite").parquet(EVENTS_OUTPUT)
journeys.write.mode("overwrite").parquet(JOURNEYS_OUTPUT)
cohorts.write.mode("overwrite").parquet(COHORT_OUTPUT)

# Confirm each saved output matches its validated in-memory population.
require_equal(spark.read.parquet(EVENTS_OUTPUT).count(), c.count() + p.count(), "persisted clinical events")
require_equal(spark.read.parquet(JOURNEYS_OUTPUT).count(), index.count(), "persisted patient journeys")
require_equal(spark.read.parquet(COHORT_OUTPUT).count(), cohorts.count(), "persisted disease cohorts")
print("SUCCESS: patient disease journey datasets published")
