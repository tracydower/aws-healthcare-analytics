# Reusable healthcare claim validation rules with configurable severity levels.
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

Severity = Literal["warning", "error"]

class Claim(BaseModel):
    claim_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    service_date: date
    paid_amount: float

class Violation(BaseModel):
    rule_id: str
    severity: Severity
    field: str
    message: str

class ValidationResult(BaseModel):
    valid: bool
    violations: list[Violation]

DEFAULT_SEVERITIES: dict[str, Severity] = {
    "NEGATIVE_PAYMENT_REVIEW": "warning",
    "FUTURE_SERVICE_DATE": "error"
}

def validate_claim_record(
    claim: Claim,
    severities: dict[str, Severity] | None = None
) -> ValidationResult:
    config = DEFAULT_SEVERITIES | (severities or {})
    violations = []

    if claim.paid_amount < 0:
        violations.append(Violation(
            rule_id="NEGATIVE_PAYMENT_REVIEW",
            severity=config["NEGATIVE_PAYMENT_REVIEW"],
            field="paid_amount",
            message="Negative payment may represent an adjustment or recoupment."
        ))

    if claim.service_date > date.today():
        violations.append(Violation(
            rule_id="FUTURE_SERVICE_DATE",
            severity=config["FUTURE_SERVICE_DATE"],
            field="service_date",
            message="Service date cannot be in the future."
        ))

    return ValidationResult(
        valid=not any(v.severity == "error" for v in violations),
        violations=violations
    )