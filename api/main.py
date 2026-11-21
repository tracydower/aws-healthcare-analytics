# FastAPI service for healthcare claim validation.
from datetime import date
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Healthcare Data Quality API",
    description="Reusable validation and monitoring services for healthcare data pipelines.",
    version="0.2.0"
)

class Claim(BaseModel):
    claim_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    service_date: date
    paid_amount: float

class Violation(BaseModel):
    rule_id: str
    severity: Literal["warning", "error"]
    field: str
    message: str

class ValidationResult(BaseModel):
    valid: bool
    violations: list[Violation]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/validate/claim", response_model=ValidationResult)
def validate_claim(claim: Claim):
    violations = []
    if claim.paid_amount < 0:
        violations.append(Violation(
            rule_id="NEGATIVE_PAYMENT_REVIEW",
            severity="warning",
            field="paid_amount",
            message="Negative payment may represent an adjustment or recoupment."
        ))
    if claim.service_date > date.today():
        violations.append(Violation(
            rule_id="FUTURE_SERVICE_DATE",
            severity="error",
            field="service_date",
            message="Service date cannot be in the future."
        ))
    return ValidationResult(
        valid=not any(v.severity == "error" for v in violations),
        violations=violations
    )