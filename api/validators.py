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

 
# Apply configurable validation rules to a healthcare claim.
def validate_claim_record(
    claim: Claim,
    severities: dict[str, Severity] | None = None
) -> ValidationResult:
    from api.rule_config import load_rules

    rules = load_rules()
    if severities:
        for rule_id, severity in severities.items():
            if rule_id in rules:
                rules[rule_id].severity = severity

    violations = []

    negative_rule = rules["NEGATIVE_PAYMENT_REVIEW"]
    if negative_rule.enabled and claim.paid_amount < negative_rule.threshold:
        violations.append(Violation(
            rule_id="NEGATIVE_PAYMENT_REVIEW",
            severity=negative_rule.severity,
            field="paid_amount",
            message="Negative payment may represent an adjustment or recoupment."
        ))

    future_rule = rules["FUTURE_SERVICE_DATE"]
    if future_rule.enabled and claim.service_date > date.today():
        violations.append(Violation(
            rule_id="FUTURE_SERVICE_DATE",
            severity=future_rule.severity,
            field="service_date",
            message="Service date cannot be in the future."
        ))

    return ValidationResult(
        valid=not any(v.severity == "error" for v in violations),
        violations=violations
    )

 