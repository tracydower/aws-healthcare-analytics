# FastAPI endpoints for reusable healthcare data-quality validation.
from fastapi import FastAPI
from api.validators import Claim, Violation, ValidationResult, validate_claim_record

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
    total_claims: int
    valid_claims: int
    invalid_claims: int
    warning_count: int
    results: list[BatchRecordResult]

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

    return BatchValidationResult(
        total_claims=len(results),
        valid_claims=sum(r.valid for r in results),
        invalid_claims=sum(not r.valid for r in results),
        warning_count=sum(
            v.severity == "warning"
            for r in results
            for v in r.violations
        ),
        results=results
    )