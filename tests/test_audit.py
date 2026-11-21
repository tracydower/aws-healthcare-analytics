# Verify audit events contain correct counts, unique IDs, and no patient-level data.
import json
import logging
from uuid import UUID
from unittest.mock import patch
from api.audit import log_validation_run

def test_audit_event(caplog):
    with caplog.at_level(logging.INFO, logger="healthcare_data_quality"):
        with patch("api.audit.logger.info") as mock_log:
            run_id = log_validation_run(4, 2, 2, 1)

    event = json.loads(mock_log.call_args.args[0])
    assert event["event"] == "claim_batch_validation"
    assert event["run_id"] == run_id
    assert event["total_records"] == 4
    assert event["valid_records"] == 2
    assert event["invalid_records"] == 2
    assert event["warning_count"] == 1
    assert event["timestamp_utc"]
    UUID(run_id)

def test_unique_run_ids():
    with patch("api.audit.logger.info"):
        first = log_validation_run(1, 1, 0, 0)
        second = log_validation_run(1, 1, 0, 0)
    assert first != second

def test_audit_does_not_log_patient_identifiers():
    with patch("api.audit.logger.info") as mock_log:
        log_validation_run(1, 1, 0, 0)
    event = json.loads(mock_log.call_args.args[0])
    assert "patient_id" not in event
    assert "claim_id" not in event