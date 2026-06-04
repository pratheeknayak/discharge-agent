"""
part2/metrics.py
================
Edit distance metrics for Part 2 learning loop.
Measures how much the doctor had to edit the agent's draft.
Lower = better (fewer corrections needed).
"""

import json

try:
    from Levenshtein import distance as lev_distance
except ImportError:
    def lev_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[:], i
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[j] = min(dp[j]+1, dp[j-1]+1, prev[j-1]+cost)
        return dp[n]


SCORED_SECTIONS = [
    "patient_demographics", "admission_discharge_dates",
    "principal_diagnosis", "secondary_diagnoses",
    "hospital_course", "procedures",
    "discharge_medications", "admission_medications",
    "medication_changes", "allergies", "lab_results",
    "pending_results", "follow_up_instructions",
    "discharge_condition",
]


def compute_edit_distance(draft: dict, edited: dict) -> float:
    """
    Normalised edit distance between two summary dicts.
    Returns float [0, 1]. Lower = better.
    """
    d = json.dumps(draft,  sort_keys=True)
    e = json.dumps(edited, sort_keys=True)
    raw = lev_distance(d, e)
    return round(raw / max(len(d), len(e), 1), 4)


def compute_section_scores(draft: dict, edited: dict) -> dict:
    """
    Per-section accuracy (1 = no edits needed, 0 = full rewrite).
    """
    scores = {}
    for sec in SCORED_SECTIONS:
        d_val = json.dumps(draft.get(sec,  ""), sort_keys=True)
        e_val = json.dumps(edited.get(sec, ""), sort_keys=True)
        if d_val == e_val:
            scores[sec] = 1.0
        else:
            raw = lev_distance(d_val, e_val)
            scores[sec] = round(1.0 - raw / max(len(d_val), len(e_val), 1), 4)
    return scores


def generate_improvement_report(metrics_list: list) -> dict:
    """Generate improvement curve from list of metrics."""
    if not metrics_list:
        return {"error": "No metrics yet"}

    curve = [
        {"patient_id": m["patient_id"],
         "edit_distance": m["edit_distance"]}
        for m in metrics_list
    ]

    eds = [m["edit_distance"] for m in metrics_list]
    improvement = (
        round(eds[0] - eds[-1], 4) if len(eds) >= 2 else 0.0
    )

    section_totals = {s: [] for s in SCORED_SECTIONS}
    for m in metrics_list:
        for sec, score in m.get("section_scores", {}).items():
            if sec in section_totals:
                section_totals[sec].append(score)

    section_averages = {
        sec: round(sum(vals)/len(vals), 4)
        for sec, vals in section_totals.items() if vals
    }

    return {
        "curve"            : curve,
        "total_improvement": improvement,
        "avg_edit_distance": round(sum(eds)/len(eds), 4),
        "best_section"     : max(section_averages,
                                  key=section_averages.get,
                                  default="N/A"),
        "worst_section"    : min(section_averages,
                                  key=section_averages.get,
                                  default="N/A"),
        "section_averages" : section_averages,
    }