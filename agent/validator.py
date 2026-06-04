"""
agent/validator.py
==================
Pydantic validation for discharge summary output.
Normalizes raw dict to EvidenceField format.
Auto-flags missing critical sections.
"""

from schemas.summary import (
    DischargeSummary, EvidenceField,
    ALL_SECTIONS, MISSING_FIELD, MISSING
)
from tools.clinical_tools import flag_for_clinician


def validate_summary(draft: dict) -> dict:
    """
    Validate and normalize discharge summary draft.

    1. Converts plain strings to EvidenceField format
    2. Fills missing sections with MISSING template
    3. Runs Pydantic validation (auto-flags critical fields)
    4. Returns validated dict

    Args:
        draft: raw dict from LLM

    Returns:
        validated dict with _validated flag
    """
    try:
        # Normalize all fields to EvidenceField format
        normalized = {}

        for key, val in draft.items():
            # Skip meta/flag fields
            if key.startswith("_") or key == "clinician_flags":
                normalized[key] = val
                continue

            if isinstance(val, dict) and "value" in val:
                # Already in EvidenceField format
                normalized[key] = val
            elif isinstance(val, str):
                normalized[key] = {
                    "value" : val.strip() if val.strip() else MISSING,
                    "source": "extracted"
                }
            elif isinstance(val, list):
                normalized[key] = {
                    "value" : "; ".join(str(v) for v in val) or MISSING,
                    "source": "extracted"
                }
            else:
                normalized[key] = {
                    "value" : str(val) if val else MISSING,
                    "source": "extracted"
                }

        # Ensure all required sections exist
        schema_input = {}
        for field in DischargeSummary.model_fields.keys():
            if field in ("clinician_flags",):
                continue
            if field in normalized:
                schema_input[field] = normalized[field]
            else:
                schema_input[field] = dict(MISSING_FIELD)

        # Run Pydantic validation
        validated_model = DischargeSummary(**schema_input)
        result = validated_model.model_dump()

        # Restore meta fields
        for key in ["_meta", "_validated", "clinician_flags"]:
            if key in draft:
                result[key] = draft[key]

        result["_validated"] = True
        return result

    except Exception as e:
        flag_for_clinician(
            reason   = "PYDANTIC VALIDATION FAILED",
            section  = "summary",
            details  = f"Validation error: {e}",
            severity = "WARNING"
        )
        draft["_validated"] = False
        draft["_validation_error"] = str(e)
        return draft