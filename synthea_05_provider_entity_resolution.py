"""AWS Glue 6.0: synthetic-label provider entity resolution with Spark ML.

Reads processed Synthea providers; creates controlled positive and negative
training pairs; evaluates on held-out provider identities; and writes ranked
real-record candidate pairs for human review. No real duplicates are assumed.
"""

import json
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BUCKET = "healthcare-analytics-td-2026"
INPUT = f"s3://{BUCKET}/processed/providers/"
OUTPUT = f"s3://{BUCKET}/curated/provider_entity_resolution/"
FEATURES = [
    "name_similarity", "address_similarity", "city_match", "zip5_match",
    "specialty_match", "geo_closeness", "npi_match",
]

spark = SparkSession.builder.appName("synthea_05_provider_entity_resolution").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def normalize(col):
    """Normalize text while removing Synthea's embedded numeric name suffixes."""
    return F.trim(F.regexp_replace(F.regexp_replace(F.upper(F.coalesce(col, F.lit(""))), r"[^A-Z ]", " "), r"\s+", " "))


def tidy_address(col):
    """Standardize common street abbreviations for feature comparison."""
    v = F.trim(F.regexp_replace(F.regexp_replace(F.upper(F.coalesce(col, F.lit(""))), r"[^A-Z0-9 ]", " "), r"\s+", " "))
    v = F.regexp_replace(v, r"\bSTREET\b", "ST")
    v = F.regexp_replace(v, r"\bROAD\b", "RD")
    v = F.regexp_replace(v, r"\bAVENUE\b", "AVE")
    v = F.regexp_replace(v, r"\bBOULEVARD\b", "BLVD")
    return v


def similarity(a, b):
    """Normalized Levenshtein similarity in [0,1], or zero for missing text."""
    longest = F.greatest(F.length(a), F.length(b))
    return F.when(longest == 0, F.lit(0.0)).otherwise(
        F.greatest(F.lit(0.0), F.lit(1.0) - F.levenshtein(a, b) / longest)
    )


def add_features(pairs):
    """Compute identical features for both synthetic and real candidate pairs."""
    lat1, lat2 = F.radians(F.col("a_lat")), F.radians(F.col("b_lat"))
    lon1, lon2 = F.radians(F.col("a_lon")), F.radians(F.col("b_lon"))
    hav = (F.pow(F.sin((lat2 - lat1) / 2), 2)
           + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin((lon2 - lon1) / 2), 2))
    km = 6371.0 * 2 * F.asin(F.sqrt(F.least(F.lit(1.0), hav)))
    return (pairs
        .withColumn("name_similarity", similarity(F.col("a_name"), F.col("b_name")))
        .withColumn("address_similarity", similarity(F.col("a_address"), F.col("b_address")))
        .withColumn("city_match", (F.col("a_city") == F.col("b_city")).cast("double"))
        .withColumn("zip5_match", ((F.col("a_zip5") != "") & (F.col("a_zip5") == F.col("b_zip5"))).cast("double"))
        .withColumn("specialty_match", ((F.col("a_specialty") != "") & (F.col("a_specialty") == F.col("b_specialty"))).cast("double"))
        .withColumn("geo_closeness", F.when(km.isNull(), 0.0).otherwise(1.0 / (1.0 + km)))
        .withColumn("npi_match", ((F.col("a_npi") != "") & (F.col("a_npi") == F.col("b_npi"))).cast("double"))
        .fillna(0.0, subset=FEATURES))


# Read source data without inferring real-world duplicate identities.
raw = spark.read.parquet(INPUT)
required = {"id", "source_dataset", "name", "address", "city", "state", "zip", "speciality", "lat", "lon", "npi"}
missing = sorted(required - set(raw.columns))
if missing:
    raise ValueError(f"Missing required provider columns: {missing}")

# Establish unique source-specific provider keys and standardized features.
providers = (raw.select(
        F.concat_ws(":", "source_dataset", "id").alias("provider_key"),
        F.col("source_dataset"), F.col("id").alias("provider_id"),
        normalize(F.col("name")).alias("name"),
        tidy_address(F.col("address")).alias("address"),
        normalize(F.col("city")).alias("city"),
        normalize(F.col("state")).alias("state"),
        F.regexp_extract(F.coalesce(F.col("zip"), F.lit("")), r"^(\d{5})", 1).alias("zip5"),
        normalize(F.col("speciality")).alias("specialty"),
        F.coalesce(F.col("npi"), F.lit("")).alias("npi"),
        F.col("lat").cast("double").alias("lat"),
        F.col("lon").cast("double").alias("lon"))
    .dropDuplicates(["provider_key"]))

# Fail if source keys were null or collide, rather than masking input problems.
source_count = raw.count()
unique_count = providers.count()
if source_count != unique_count or providers.filter(F.col("provider_key").isNull()).limit(1).count():
    raise ValueError(f"Provider keys invalid: source_rows={source_count}, unique_keys={unique_count}")
print(f"PROVIDERS={unique_count}")

# Make two deterministically altered versions of each provider for known positives.
# Variation A models initials and common street abbreviations.
variation_a = (providers
    .withColumn("variant", F.lit("initial_and_street"))
    .withColumn("variant_name", F.regexp_replace(F.col("name"), r"^([A-Z])[A-Z]+ ", "$1 "))
    .withColumn("variant_address", F.regexp_replace(F.col("address"), r"\bST\b", "STREET")))
# Variation B models a missing first character and missing specialty.
variation_b = (providers
    .withColumn("variant", F.lit("name_typo_and_missing_specialty"))
    .withColumn("variant_name", F.when(F.length("name") > 5, F.expr("substring(name, 2, length(name))")).otherwise(F.col("name")))
    .withColumn("variant_address", F.col("address")))
variants = variation_a.unionByName(variation_b)

positive = variants.select(
    F.col("provider_key").alias("left_key"), F.col("provider_key").alias("right_key"),
    F.col("name").alias("a_name"), F.col("variant_name").alias("b_name"),
    F.col("address").alias("a_address"), F.col("variant_address").alias("b_address"),
    F.col("city").alias("a_city"), F.col("city").alias("b_city"),
    F.col("zip5").alias("a_zip5"), F.col("zip5").alias("b_zip5"),
    F.col("specialty").alias("a_specialty"),
    F.when(F.col("variant") == "name_typo_and_missing_specialty", F.lit("")).otherwise(F.col("specialty")).alias("b_specialty"),
    F.col("npi").alias("a_npi"), F.col("npi").alias("b_npi"),
    F.col("lat").alias("a_lat"), F.col("lat").alias("b_lat"),
    F.col("lon").alias("a_lon"), F.col("lon").alias("b_lon"),
    F.lit(1.0).alias("label"), F.lit("synthetic_positive").alias("pair_origin"))

# Candidate generation: shared geographic coordinate or identical five-digit ZIP.
# Same-name records with different IDs are excluded from synthetic negatives:
# their true identities are unknown, so they cannot be confidently labeled.
a = providers.alias("a")
b = providers.alias("b")
pairs = (a.join(b,
    (F.col("a.provider_key") < F.col("b.provider_key")) &
    ((F.col("a.lat") == F.col("b.lat")) & (F.col("a.lon") == F.col("b.lon")) |
     ((F.col("a.zip5") != "") & (F.col("a.zip5") == F.col("b.zip5"))) |
     ((F.col("a.name") != "") & (F.col("a.name") == F.col("b.name")))), "inner")
    .select(
        F.col("a.provider_key").alias("left_key"), F.col("b.provider_key").alias("right_key"),
        *[F.col(f"a.{k}").alias(f"a_{k}") for k in ["name", "address", "city", "zip5", "specialty", "npi", "lat", "lon"]],
        *[F.col(f"b.{k}").alias(f"b_{k}") for k in ["name", "address", "city", "zip5", "specialty", "npi", "lat", "lon"]]))

# For negative training, limit high-volume facilities to five partners per left ID.
w = Window.partitionBy("left_key").orderBy(F.xxhash64("left_key", "right_key"))
negative = (pairs.filter(F.col("a_name") != F.col("b_name"))
    .withColumn("candidate_rank", F.row_number().over(w))
    .filter(F.col("candidate_rank") <= 5).drop("candidate_rank")
    .withColumn("label", F.lit(0.0))
    .withColumn("pair_origin", F.lit("assumed_negative_distinct_names")))

# Hold out left provider identities using deterministic hashing before fitting.
# This prevents a provider's synthetic variants appearing in both splits.
examples = add_features(positive.unionByName(negative)).withColumn(
    "split", F.when(F.pmod(F.xxhash64("left_key"), F.lit(5)) == 0, "test").otherwise("train"))
assembler = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="keep")
feature_data = assembler.transform(examples)
train = feature_data.filter(F.col("split") == "train").select("features", "label")
test = feature_data.filter(F.col("split") == "test").select("features", "label", "pair_origin")

# Train using Spark ML; no scikit-learn installation required in Glue.
model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=60, regParam=0.05)
fit = model.fit(train)
scored_test = fit.transform(test).withColumn("match_probability", vector_to_array("probability")[1])

# Evaluate classification at an explicit, reproducible probability threshold.
threshold = 0.50
scored_test = scored_test.withColumn("predicted_match", (F.col("match_probability") >= threshold).cast("int"))
stats = scored_test.agg(
    F.sum(F.when((F.col("label") == 1) & (F.col("predicted_match") == 1), 1).otherwise(0)).alias("tp"),
    F.sum(F.when((F.col("label") == 0) & (F.col("predicted_match") == 1), 1).otherwise(0)).alias("fp"),
    F.sum(F.when((F.col("label") == 0) & (F.col("predicted_match") == 0), 1).otherwise(0)).alias("tn"),
    F.sum(F.when((F.col("label") == 1) & (F.col("predicted_match") == 0), 1).otherwise(0)).alias("fn"),
).first().asDict()
tp, fp, tn, fn = (int(stats[k] or 0) for k in ("tp", "fp", "tn", "fn"))
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(fit.transform(test))
metrics = {"note": "Synthetic perturbation evaluation; not validated real-world duplicate detection",
           "providers": unique_count, "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}
print("SYNTHETIC_TEST_METRICS=" + json.dumps(metrics, sort_keys=True))

# Score real distinct-ID candidate records, never automatically merge records.
real_scores = (fit.transform(assembler.transform(add_features(pairs)))
    .withColumn("match_probability", vector_to_array("probability")[1])
    .select("left_key", "right_key", "a_name", "b_name", "a_address", "b_address",
            "a_city", "b_city", "a_zip5", "b_zip5", "a_specialty", "b_specialty",
            "name_similarity", "address_similarity", "city_match", "zip5_match",
            "specialty_match", "geo_closeness", "npi_match", "match_probability")
    # Record-level decision explicitly supplied by the project owner.
    .withColumn("review_status", F.when(
        ((F.col("left_key") == "synthea_1k_2021:f591810d-5443-31c5-8089-e1afe233eeca") &
         (F.col("right_key") == "synthea_sample_data_csv_latest:7ba0d69a-a093-3dbd-a25d-53c91e1ffbf2")) |
        ((F.col("right_key") == "synthea_1k_2021:f591810d-5443-31c5-8089-e1afe233eeca") &
         (F.col("left_key") == "synthea_sample_data_csv_latest:7ba0d69a-a093-3dbd-a25d-53c91e1ffbf2")),
        "KNOWN_NONMATCH").otherwise("UNVERIFIED_CANDIDATE_ONLY")))

# Save the scored pairs and evaluation artifacts. Outputs are never called confirmed matches.
real_scores.write.mode("overwrite").parquet(OUTPUT + "review_candidates/")
spark.createDataFrame([metrics]).coalesce(1).write.mode("overwrite").json(OUTPUT + "synthetic_evaluation/")
print(f"WROTE review candidates to {OUTPUT}review_candidates/")
print(f"WROTE synthetic evaluation to {OUTPUT}synthetic_evaluation/")
print("SUCCESS: No real provider records were merged or assigned confirmed-match status.")
