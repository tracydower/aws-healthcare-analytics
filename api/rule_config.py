# Load and validate configurable healthcare data-quality rules.
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class RuleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    severity: Literal["warning", "error"]
    threshold: float | None = Field(default=None)

def load_rules() -> dict[str, RuleSettings]:
    path = Path(__file__).with_name("rules.json")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {name: RuleSettings.model_validate(settings) for name, settings in raw.items()}


