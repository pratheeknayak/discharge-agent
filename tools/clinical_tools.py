"""
tools/clinical_tools.py — Fixed version
=========================================
Fix: compare_medications now handles structured medication objects
     not just plain strings
"""

from collections import defaultdict
from typing import List

_flags: List[dict] = []

def reset_flags():
    _flags.clear()

def get_all_flags() -> List[dict]:
    return list(_flags)


def detect_conflicts(extractions: List[dict]) -> dict:
    try:
        by_field = defaultdict(list)
        for e in extractions:
            by_field[e["field"]].append({
                "source": e["source"],
                "value" : str(e["value"])
            })
        conflicts = {
            field: values
            for field, values in by_field.items()
            if len({v["value"].lower().strip()
                    for v in values}) > 1
        }
        return {
            "success"  : True,
            "conflicts": conflicts,
            "summary"  : (
                f"Found {len(conflicts)} conflict(s)"
                if conflicts else "No conflicts"
            )
        }
    except Exception as e:
        return {"success": False, "conflicts": {}, "error": str(e)}


def _normalize_med(med) -> str:
    """
    Convert any medication representation to a clean name string.
    Handles: str, dict, list
    """
    if isinstance(med, str):
        return med.strip()
    elif isinstance(med, dict):
        # Extract name from structured object
        return med.get("name", str(med)).strip()
    elif isinstance(med, list):
        # Join list items
        return "; ".join(_normalize_med(m) for m in med)
    else:
        return str(med).strip()


def compare_medications(
    admission_meds: List,
    discharge_meds: List
) -> dict:
    """
    Compare admission vs discharge medications.
    Handles plain strings AND structured medication dicts.
    """
    try:
        # Normalize to clean name strings
        adm_names = {
            _normalize_med(m).lower()
            for m in admission_meds
            if _normalize_med(m) and len(_normalize_med(m)) > 2
        }
        dis_names = {
            _normalize_med(m).lower()
            for m in discharge_meds
            if _normalize_med(m) and len(_normalize_med(m)) > 2
        }

        # Remove empty/junk entries
        adm_names = {
            n for n in adm_names
            if n and "[missing" not in n and n not in
            {"the", "and", "for", "with", "tab", "inj", "cap"}
        }
        dis_names = {
            n for n in dis_names
            if n and "[missing" not in n and n not in
            {"the", "and", "for", "with", "tab", "inj", "cap"}
        }

        added     = sorted(dis_names - adm_names)
        stopped   = sorted(adm_names - dis_names)
        continued = sorted(adm_names & dis_names)

        changes = []
        for med in added:
            changes.append({
                "medication" : med,
                "change_type": "ADDED",
                "flagged"    : True,
                "reason"     : "not documented"
            })
        for med in stopped:
            changes.append({
                "medication" : med,
                "change_type": "STOPPED",
                "flagged"    : True,
                "reason"     : "not documented"
            })

        return {
            "success"  : True,
            "added"    : added,
            "stopped"  : stopped,
            "continued": continued,
            "changes"  : changes,
            "summary"  : (
                f"Added: {len(added)}, "
                f"Stopped: {len(stopped)}, "
                f"Continued: {len(continued)}"
            )
        }
    except Exception as e:
        return {
            "success": False, "changes": [],
            "summary": f"Reconciliation failed: {e}",
            "error": str(e)
        }


_INTERACTIONS = {
    ("warfarin",      "aspirin"):    "HIGH: Increased bleeding risk",
    ("warfarin",      "ibuprofen"):  "HIGH: Increased bleeding risk",
    ("metformin",     "contrast"):   "MODERATE: Hold before contrast",
    ("ssri",          "tramadol"):   "HIGH: Serotonin syndrome risk",
    ("lisinopril",    "potassium"):  "MODERATE: Hyperkalemia risk",
    ("insulin",       "alcohol"):    "HIGH: Severe hypoglycemia risk",
    ("meropenem",     "valproate"):  "HIGH: Meropenem reduces valproate",
    ("lantus",        "actrapid"):   "LOW: Dual insulin — monitor glucose",
    ("ciprofloxacin", "insulin"):    "MODERATE: May alter glucose control",
}

def check_drug_interactions(medications: List) -> dict:
    try:
        flags  = []
        lowers = [_normalize_med(m).lower() for m in medications]
        lowers = [m for m in lowers if m and len(m) > 3]
        for i, m1 in enumerate(lowers):
            for m2 in lowers[i + 1:]:
                for (a, b), sev in _INTERACTIONS.items():
                    if (a in m1 and b in m2) or (a in m2 and b in m1):
                        flags.append({
                            "drug1"   : m1,
                            "drug2"   : m2,
                            "severity": sev
                        })
        return {
            "success"     : True,
            "interactions": flags,
            "summary"     : (
                f"Found {len(flags)} interaction(s)"
                if flags else "No known interactions found"
            )
        }
    except Exception as e:
        return {"success": False, "interactions": [], "error": str(e)}


def flag_for_clinician(
    reason: str, section: str,
    details: str, severity: str = "REVIEW"
) -> dict:
    entry = {
        "section" : section,
        "reason"  : reason,
        "details" : details,
        "severity": severity.upper()
    }
    _flags.append(entry)
    return {"success": True, "flagged": entry}