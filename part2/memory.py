"""
part2/memory.py
================
Correction memory store.
Stores (draft, edited) pairs and injects lessons into future prompts.
"""

import os
import json
from datetime import datetime

MEMORY_FILE = "outputs/correction_memory.json"


def _load() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def _save(memory: list):
    os.makedirs("outputs", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def store_correction(
    patient_id: str, draft: dict, edited: dict,
    edit_distance: float, section_scores: dict
):
    memory = _load()
    changes = []
    for key in edited:
        if key.startswith("_"):
            continue
        d_val = str(draft.get(key,  ""))
        e_val = str(edited.get(key, ""))
        if d_val != e_val:
            changes.append({
                "section": key,
                "draft"  : d_val[:200],
                "edited" : e_val[:200],
            })
    memory.append({
        "patient_id"    : patient_id,
        "timestamp"     : datetime.now().isoformat(),
        "edit_distance" : edit_distance,
        "section_scores": section_scores,
        "changes"       : changes,
    })
    _save(memory)
    print(f"  [Memory] Stored correction for {patient_id} "
          f"(edit_distance={edit_distance:.3f})")


def build_memory_injection(max_examples: int = 3) -> str:
    memory = _load()
    if not memory:
        return ""

    recent = memory[-max_examples:]
    lines  = [
        "LEARNING FROM PAST DOCTOR CORRECTIONS:",
        "Apply these standards proactively:\n"
    ]

    seen = set()
    for entry in recent:
        ed = entry.get("edit_distance", 1.0)
        lines.append(
            f"[{entry['patient_id']} | ed={ed:.3f}]"
        )
        for change in entry.get("changes", []):
            key = f"{change['section']}:{change['draft'][:40]}"
            if key not in seen:
                seen.add(key)
                lines.append(f"  Section '{change['section']}':")
                lines.append(f"    Was : {change['draft'][:100]}")
                lines.append(f"    Now : {change['edited'][:100]}")
        lines.append("")

    return "\n".join(lines)


def get_all_metrics() -> list:
    return [
        {
            "patient_id"    : e["patient_id"],
            "timestamp"     : e["timestamp"],
            "edit_distance" : e["edit_distance"],
            "section_scores": e.get("section_scores", {}),
        }
        for e in _load()
    ]