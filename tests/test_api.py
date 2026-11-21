# Automated tests for healthcare API health checks and claim validation.
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_valid_claim():
    response = client.post("/api/v1/validate/claim", json={
        "claim_id": "CLM001",
        "patient_id": "PAT123",
        "service_date": "2025-06-15",
        "paid_amount": 250.00
    })
    assert response.status_code == 200
    assert response.json() == {"valid": True, "violations": []}

def test_negative_payment():
    response = client.post("/api/v1/validate/claim", json={
        "claim_id": "CLM002",
        "patient_id": "PAT123",
        "service_date": "2025-06-15",
        "paid_amount": -75.00
    })
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["violations"][0]["rule_id"] == "NEGATIVE_PAYMENT_REVIEW"

def test_future_service_date():
    response = client.post("/api/v1/validate/claim", json={
        "claim_id": "CLM003",
        "patient_id": "PAT123",
        "service_date": "2099-01-01",
        "paid_amount": 100.00
    })
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["violations"][0]["rule_id"] == "FUTURE_SERVICE_DATE"

def test_missing_patient_id():
    response = client.post("/api/v1/validate/claim", json={
        "claim_id": "CLM004",
        "service_date": "2025-06-15",
        "paid_amount": 100.00
    })
    assert response.status_code == 422


    # Test batch validation with mixed results, warnings, and empty input.
def test_batch_mixed_claims():
    claims = [
        {"claim_id": "C001", "patient_id": "P001", "service_date": "2025-06-15", "paid_amount": 100},
        {"claim_id": "C002", "patient_id": "P002", "service_date": "2099-01-01", "paid_amount": 200},
        {"claim_id": "C003", "patient_id": "P003", "service_date": "2025-06-15", "paid_amount": -50}
    ]
    response = client.post("/api/v1/validate/claims", json=claims)
    assert response.status_code == 200
    result = response.json()
    assert result["total_claims"] == 3
    assert result["valid_claims"] == 2
    assert result["invalid_claims"] == 1
    assert result["warning_count"] == 1

def test_batch_empty():
    response = client.post("/api/v1/validate/claims", json=[])
    assert response.status_code == 200
    assert response.json()["total_claims"] == 0

# Verify malformed records are reported without rejecting the entire batch.
def test_batch_invalid_schema():
    response = client.post("/api/v1/validate/claims", json=[
        {"claim_id": "C004", "service_date": "2025-06-15", "paid_amount": 100}
    ])
    assert response.status_code == 200
    result = response.json()
    assert result["total_claims"] == 1
    assert result["invalid_claims"] == 1
    assert result["results"][0]["row_number"] == 1
    assert result["results"][0]["violations"][0]["rule_id"] == "SCHEMA_VALIDATION_ERROR"


    # Verify valid, malformed, and warning-only claims are processed independently.
def test_batch_record_isolation():
    claims = [
        {"claim_id": "C001", "patient_id": "P001", "service_date": "2025-06-15", "paid_amount": 100},
        {"claim_id": "C002", "service_date": "2025-06-15", "paid_amount": 200},
        {"claim_id": "C003", "patient_id": "P003", "service_date": "2025-06-15", "paid_amount": -50},
        {"claim_id": "C004", "patient_id": "P004", "service_date": "2099-01-01", "paid_amount": 100}
    ]
    response = client.post("/api/v1/validate/claims", json=claims)
    assert response.status_code == 200
    result = response.json()
    assert result["total_claims"] == 4
    assert result["valid_claims"] == 2
    assert result["invalid_claims"] == 2
    assert result["warning_count"] == 1
    assert [r["row_number"] for r in result["results"]] == [1, 2, 3, 4]
    assert result["results"][1]["violations"][0]["rule_id"] == "SCHEMA_VALIDATION_ERROR"
    assert result["results"][2]["violations"][0]["rule_id"] == "NEGATIVE_PAYMENT_REVIEW"
    assert result["results"][3]["violations"][0]["rule_id"] == "FUTURE_SERVICE_DATE"


    # Verify rule configuration controls severity and enabled status.
from unittest.mock import patch
from api.rule_config import RuleSettings

def test_configurable_severity():
    rules = {
        "NEGATIVE_PAYMENT_REVIEW": RuleSettings(enabled=True, severity="error", threshold=0),
        "FUTURE_SERVICE_DATE": RuleSettings(enabled=True, severity="error")
    }
    with patch("api.rule_config.load_rules", return_value=rules):
        response = client.post("/api/v1/validate/claim", json={
            "claim_id": "C005",
            "patient_id": "P005",
            "service_date": "2025-06-15",
            "paid_amount": -50
        })
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["violations"][0]["severity"] == "error"

def test_disabled_rule():
    rules = {
        "NEGATIVE_PAYMENT_REVIEW": RuleSettings(enabled=False, severity="warning", threshold=0),
        "FUTURE_SERVICE_DATE": RuleSettings(enabled=True, severity="error")
    }
    with patch("api.rule_config.load_rules", return_value=rules):
        response = client.post("/api/v1/validate/claim", json={
            "claim_id": "C006",
            "patient_id": "P006",
            "service_date": "2025-06-15",
            "paid_amount": -50
        })
    assert response.status_code == 200
    assert response.json() == {"valid": True, "violations": []}


# Verify that the batch API returns a traceable audit run ID.
def test_batch_returns_audit_run_id():
    from uuid import UUID

    response = client.post("/api/v1/validate/claims", json=[
        {
            "claim_id": "C100",
            "patient_id": "P100",
            "service_date": "2025-06-15",
            "paid_amount": 100
        }
    ])

    assert response.status_code == 200
    result = response.json()
    assert "run_id" in result
    UUID(result["run_id"])
    assert result["total_claims"] == 1
    assert result["valid_claims"] == 1

    