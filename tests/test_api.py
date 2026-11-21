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