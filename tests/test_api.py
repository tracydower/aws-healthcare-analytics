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

def test_batch_invalid_schema():
    response = client.post("/api/v1/validate/claims", json=[
        {"claim_id": "C004", "service_date": "2025-06-15", "paid_amount": 100}
    ])
    assert response.status_code == 422