
# Log claim-validation audit events to CloudWatch Logs, S3, and custom CloudWatch metrics.
import json
import logging
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import boto3

logger = logging.getLogger("healthcare_data_quality")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

def publish_cloudwatch_metrics(event: dict, timestamp: datetime) -> None:
    if os.getenv("CLOUDWATCH_METRICS_ENABLED", "false").lower() != "true":
        return

    metrics = {
        "ClaimsProcessed": "total_records",
        "ValidClaims": "valid_records",
        "InvalidClaims": "invalid_records",
        "ValidationWarnings": "warning_count"
    }

    try:
        boto3.client("cloudwatch").put_metric_data(
            Namespace="Healthcare/DataQuality",
            MetricData=[
                {
                    "MetricName": name,
                    "Dimensions": [
                        {"Name": "FunctionName", "Value": "healthcare-data-quality-api"}
                    ],
                    "Timestamp": timestamp,
                    "Value": event[field],
                    "Unit": "Count"
                }
                for name, field in metrics.items()
            ]
        )
    except Exception:
        logger.exception("CloudWatch metric publishing failed")

def log_validation_run(total: int, valid: int, invalid: int, warnings: int) -> str:
    run_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)

    event = {
        "event": "claim_batch_validation",
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "total_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "warning_count": warnings
    }

    logger.info(json.dumps(event))

    bucket = os.getenv("AUDIT_S3_BUCKET")
    if bucket:
        key = (
            f"audit/claim_validation/"
            f"{timestamp:%Y/%m/%d}/{run_id}.json"
        )
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(event).encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256"
        )

    publish_cloudwatch_metrics(event, timestamp)

    return run_id
