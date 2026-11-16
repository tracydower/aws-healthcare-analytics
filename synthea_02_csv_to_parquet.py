"""AWS Glue PySpark job: combine two Synthea CSV releases into curated Parquet tables.

Scope: the 17 tables other than patients. The already validated patients Parquet
is intentionally NOT overwritten. Raw CSV files are never modified.

To run as an AWS Glue Spark ETL job, configure a Glue job with this script,
Glue 6.0, and an IAM role permitted to read/write the project S3 bucket.
"""

from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Start the Glue Spark session supplied to this job.
sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

# Set source and destination paths for this specific portfolio project.
BUCKET = "s3://healthcare-analytics-td-2026"
SOURCE_OLD = "synthea_1k_2021"
SOURCE_NEW = "synthea_sample_data_csv_latest"

# Keep patients untouched because its Parquet output was already validated.
TABLES = [
    "allergies", "careplans", "claims", "claims_transactions", "conditions",
    "devices", "encounters", "imaging_studies", "immunizations",
    "medications", "observations", "organizations", "payer_transitions",
    "payers", "procedures", "providers", "supplies",
]

# Confirmed equivalent fields in the two payer_transitions releases.
COLUMN_RENAMES = {
    "payer_transitions": {
        "start_year": "start_date",  # The older field actually contains timestamps.
        "end_year": "end_date",      # The older field actually contains timestamps.
        "ownership": "plan_ownership",
        "ownername": "owner_name",
    },
}


def load_csv(table_name, source_name):
    """Read one CSV correctly, standardize names, and retain source lineage."""
    if source_name == SOURCE_OLD:
        path = f"{BUCKET}/raw/{table_name}/{table_name}.csv"
    else:
        path = f"{BUCKET}/raw/{SOURCE_NEW}/{table_name}.csv"

    # Explicit header handling prevents CSV header rows from becoming data.
    df = (spark.read.option("header", "true")
                     .option("inferSchema", "true")
                     .csv(path))

    # Standardize casing without inventing any business-data columns.
    for column_name in df.columns:
        df = df.withColumnRenamed(column_name, column_name.lower())

    # Rename only the mappings confirmed by inspecting source data.
    if source_name == SOURCE_OLD:
        for old_name, new_name in COLUMN_RENAMES.get(table_name, {}).items():
            df = df.withColumnRenamed(old_name, new_name)

    # Preserve the exact data release that contributed each record.
    df = df.withColumn("source_dataset", F.lit(source_name))
    return df


# Count each input and validate every write before moving to the next table.
for table in TABLES:
    print(f"\n=== Processing {table} ===", flush=True)
    df_old = load_csv(table, SOURCE_OLD)
    df_new = load_csv(table, SOURCE_NEW)

    old_rows = df_old.count()
    new_rows = df_new.count()

    # Union by column name: differences become nullable fields, not dropped data.
    # Spark promotes compatible differing types, typically to string when needed.
    df_combined = df_old.unionByName(df_new, allowMissingColumns=True)

    # Preserve the original provider metrics separately: utilization is NOT
    # assumed to equal encounters or procedures.
    expected_rows = old_rows + new_rows
    output = f"{BUCKET}/processed/{table}/"

    # Avoid overwriting a validated output with an unexpectedly empty source.
    if old_rows == 0 or new_rows == 0:
        raise ValueError(f"{table}: one source is empty; refusing to overwrite")

    # Save in Parquet format; this changes ONLY the processed/ output path.
    df_combined.write.mode("overwrite").parquet(output)

    # Reopen from S3 and verify that the Parquet row count matches the inputs.
    written_rows = spark.read.parquet(output).count()
    if written_rows != expected_rows:
        raise RuntimeError(
            f"{table}: verification failed: expected={expected_rows}, actual={written_rows}"
        )

    print(
        f"PASS {table}: original={old_rows:,}, new={new_rows:,}, "
        f"combined={written_rows:,}, columns={len(df_combined.columns)}; "
        f"output={output}",
        flush=True,
    )

print("\nFinished 17 tables. Existing processed/patients/ was not modified.")
