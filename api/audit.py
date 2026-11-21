# Generate structured audit events and optionally persist them to Amazon S3.
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

    # Preserve structured console logging.
    logger.info(json.dumps(event))

    # Persist to S3 only when a bucket is configured.
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

    return run_id