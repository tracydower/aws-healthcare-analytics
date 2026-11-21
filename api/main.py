# FastAPI endpoints for reusable healthcare data-quality validation.
from fastapi import FastAPI
from api.validators import Claim, ValidationResult, validate_claim_record

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


# Validate multiple healthcare claims and summarize the results.
from pydantic import BaseModel

class BatchValidationResult(BaseModel):
    total_claims: int
    valid_claims: int
    invalid_claims: int
    warning_count: int
    results: list[ValidationResult]

@app.post("/api/v1/validate/claims", response_model=BatchValidationResult)
def validate_claims(claims: list[Claim]):
    results = [validate_claim_record(claim) for claim in claims]
    return BatchValidationResult(
        total_claims=len(results),
        valid_claims=sum(r.valid for r in results),
        invalid_claims=sum(not r.valid for r in results),
        warning_count=sum(
            v.severity == "warning"
            for result in results
            for v in result.violations
        ),
        results=results
    )