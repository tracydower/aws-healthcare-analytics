# FastAPI endpoints for reusable healthcare data-quality validation.
from fastapi import FastAPI
from api.validators import Claim, Violation, ValidationResult, validate_claim_record
from api.audit import log_validation_run

app = FastAPI(
    title="Healthcare Data Quality API",
    description="Reusable validation and monitoring services for healthcare data pipelines.",
    version="0.3.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/validate/claim", response_model=ValidationResult)
def validate_claim(claim: Claim):
    return validate_claim_record(claim)

# Validate claims independently so malformed records do not reject an entire batch.
from pydantic import BaseModel, ValidationError

class BatchRecordResult(BaseModel):
    row_number: int
    valid: bool
    violations: list[Violation]

class BatchValidationResult(BaseModel):
    run_id: str
    total_claims: int
    valid_claims: int
    invalid_claims: int
    warning_count: int
    results: list[BatchRecordResult]

# Validate each claim independently and record an auditable batch execution.
@app.post("/api/v1/validate/claims", response_model=BatchValidationResult)
def validate_claims(claims: list[dict]):
    results = []

    for row_number, raw_claim in enumerate(claims, start=1):
        try:
            claim = Claim.model_validate(raw_claim)
            result = validate_claim_record(claim)
            results.append(BatchRecordResult(
                row_number=row_number,
                valid=result.valid,
                violations=result.violations
            ))
        except ValidationError as exc:
            violations = [
                Violation(
                    rule_id="SCHEMA_VALIDATION_ERROR",
                    severity="error",
                    field=".".join(str(x) for x in error["loc"]),
                    message=error["msg"]
                )
                for error in exc.errors()
            ]
            results.append(BatchRecordResult(
                row_number=row_number,
                valid=False,
                violations=violations
            ))

    total = len(results)
    valid = sum(r.valid for r in results)
    invalid = total - valid
    warnings = sum(
        v.severity == "warning"
        for result in results
        for v in result.violations
    )

    # Record the execution and capture the generated audit identifier.
    run_id = log_validation_run(
        total=total,
        valid=valid,
        invalid=invalid,
        warnings=warnings
    )

    return BatchValidationResult(
        run_id=run_id,
        total_claims=total,
        valid_claims=valid,
        invalid_claims=invalid,
        warning_count=warnings,
        results=results
    )

