"""
schemas/summary.py
==================
Pydantic models for discharge summary validation.
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional

MISSING = "[MISSING - FLAG FOR REVIEW]"
PENDING = "PENDING"

ALL_SECTIONS = [
    "patient_demographics", "admission_discharge_dates",
    "principal_diagnosis",  "secondary_diagnoses",
    "hospital_course",      "procedures",
    "admission_medications","discharge_medications",
    "medication_changes",   "allergies",
    "lab_results",          "pending_results",
    "follow_up_instructions","discharge_condition",
]

MISSING_FIELD = {"value": MISSING, "source": "not_found"}


class EvidenceField(BaseModel):
    value : str
    source: str = "not_found"

    @field_validator("value")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            return MISSING
        return v.strip()

    def is_missing(self) -> bool:
        return MISSING in self.value


class ClinicianFlag(BaseModel):
    section : str
    reason  : str
    details : str
    severity: str = "REVIEW"


class DischargeSummary(BaseModel):
    patient_demographics     : EvidenceField
    admission_discharge_dates: EvidenceField
    principal_diagnosis      : EvidenceField
    secondary_diagnoses      : EvidenceField
    hospital_course          : EvidenceField
    procedures               : EvidenceField
    admission_medications    : EvidenceField
    discharge_medications    : EvidenceField
    medication_changes       : EvidenceField
    allergies                : EvidenceField
    lab_results              : EvidenceField
    pending_results          : EvidenceField
    follow_up_instructions   : EvidenceField
    discharge_condition      : EvidenceField
    clinician_flags          : List[ClinicianFlag] = []

    @model_validator(mode="after")
    def flag_missing_critical(self):
        critical = [
            "patient_demographics", "principal_diagnosis",
            "discharge_medications", "discharge_condition",
        ]
        for field_name in critical:
            field = getattr(self, field_name, None)
            if field and field.is_missing():
                self.clinician_flags.append(ClinicianFlag(
                    section  = field_name,
                    reason   = "MISSING CRITICAL FIELD",
                    details  = f"'{field_name}' not found in documents.",
                    severity = "CRITICAL"
                ))
        return self