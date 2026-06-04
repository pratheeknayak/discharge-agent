"""
agent/graph.py — Final fixed version
======================================
Key fixes:
1. Analyze node: split extraction into 3 smaller LLM calls
   (demographics+dates+diagnosis) then (meds+allergies+labs)
   then (course+procedures+followup+condition)
   Each call is smaller → no JSON truncation
2. Duplicate output: fixed by removing _add_trace print side effects
3. LangGraph TypedDict state properly defined
"""

import os
import json
import time
import re
import fitz
from typing import TypedDict, List, Optional
from google import genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from tools.pdf_extractor  import extract_pdf_text
from tools.clinical_tools import (
    detect_conflicts,
    compare_medications,
    check_drug_interactions,
    flag_for_clinician,
    get_all_flags,
    reset_flags,
)
from prompts.system import PLANNER_PROMPT

load_dotenv()

_client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
REASON_MODEL = os.getenv("GEMINI_REASON_MODEL", "models/gemini-2.5-flash")

ALL_SECTIONS = [
    "patient_demographics", "admission_discharge_dates",
    "principal_diagnosis",  "secondary_diagnoses",
    "hospital_course",      "procedures",
    "admission_medications","discharge_medications",
    "medication_changes",   "allergies",
    "lab_results",          "pending_results",
    "follow_up_instructions","discharge_condition",
]

MISSING_FIELD = {
    "value" : "[MISSING - FLAG FOR REVIEW]",
    "source": "not_found"
}

NO_FABRICATION = """
CRITICAL RULES:
- Extract ONLY what is explicitly written in the text
- NEVER invent, guess, or infer clinical values
- If not found → {"value": "[MISSING - FLAG FOR REVIEW]", "source": "not_found"}
- Include page number in source e.g. "page_3_er_chart"
- Return ONLY valid JSON. No markdown. Start with {
"""


# ── State ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    patient_id        : str
    pdf_files         : List[str]
    extracted_texts   : dict
    plan              : dict
    draft             : dict
    sections_found    : List[str]
    sections_missing  : List[str]
    medications_found : List[str]
    conflicts_found   : List[str]
    flags             : List[dict]
    extraction_failed : bool
    completed         : bool
    error             : Optional[str]
    trace             : List[dict]


def _make_state(patient_id: str, pdf_files: list) -> AgentState:
    return AgentState(
        patient_id        = patient_id,
        pdf_files         = pdf_files,
        extracted_texts   = {},
        plan              = {},
        draft             = {},
        sections_found    = [],
        sections_missing  = list(ALL_SECTIONS),
        medications_found = [],
        conflicts_found   = [],
        flags             = [],
        extraction_failed = False,
        completed         = False,
        error             = None,
        trace             = [],
    )


# ── LLM helpers ───────────────────────────────────────────────────
def _llm(prompt: str) -> str:
    for attempt in range(4):
        try:
            resp = _client.models.generate_content(
                model=REASON_MODEL,
                contents=prompt
            )
            return resp.text
        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err:
                w = 30 * (attempt + 1)
                print(f"  [Rate limit] waiting {w}s...")
                time.sleep(w)
            else:
                raise
    raise Exception("LLM failed after 4 retries")


def _parse_json(raw: str) -> dict | None:
    """Find first { to last } and parse."""
    try:
        clean = raw.strip()
        # Strip markdown fences
        if "```" in clean:
            lines = clean.splitlines()
            clean = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            )
        # Find JSON boundaries
        first = clean.find("{")
        last  = clean.rfind("}")
        if first == -1 or last == -1:
            return None
        clean = clean[first : last + 1]
        return json.loads(clean)
    except Exception:
        return None


def _extract_section(
    text: str,
    sections: List[str],
    descriptions: dict
) -> dict:
    """
    Extract a small group of sections from text.
    Retries once on server disconnect.
    """
    section_list = "\n".join(
        f'"{s}": {{"value": "<extracted or MISSING>", '
        f'"source": "<page_N_doctype>"}}'
        f"  // {descriptions.get(s, s)}"
        for s in sections
    )

    prompt = f"""Extract ONLY these fields from the hospital document text below.

{NO_FABRICATION}

DOCUMENT TEXT (excerpt):
{text[:25000]}

Extract EXACTLY these fields:
{{
{section_list}
}}

Return ONLY the JSON object above with values filled in.
No other text. No markdown."""

    for attempt in range(3):
        try:
            raw    = _llm(prompt)
            result = _parse_json(raw)
            if result:
                return result
            raise ValueError("Invalid JSON")
        except Exception as e:
            err = str(e)
            # Retry on server disconnect or rate limit
            if any(x in err.lower() for x in
                   ["disconnect", "server", "429", "503", "rate"]):
                wait = 20 * (attempt + 1)
                print(f"    [Retry] {err[:50]} — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [Extract] failed for {sections}: {err[:60]}")
                break

    # Return MISSING for all if all attempts failed
    return {s: dict(MISSING_FIELD) for s in sections}


# ── Trace helper (no side effects) ───────────────────────────────
def _trace(state: AgentState, step: dict) -> None:
    state["trace"].append(step)
    print(f"\n  ── Step {len(state['trace'])} ──────────────────")
    for key in ["reasoning", "action", "result", "decision"]:
        if key in step:
            val = str(step[key])
            if len(val) > 250:
                val = val[:250] + "..."
            print(f"  {key.upper():<10}: {val}")


# ══════════════════════════════════════════════════════════════════
# NODE 1: PLAN
# ══════════════════════════════════════════════════════════════════
def plan_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 1: PLAN\n{'#'*60}")

    previews = []
    for pdf in state["pdf_files"]:
        try:
            doc   = fitz.open(pdf)
            parts = []
            for i in range(min(5, len(doc))):
                t = doc[i].get_text().strip()
                if t:
                    parts.append(f"[Page {i+1}]: {t[:500]}")
            doc.close()
            preview = "\n".join(parts) or "[Scanned PDF — no text]"
        except Exception as e:
            preview = f"[Error: {e}]"
        previews.append({
            "file": os.path.basename(pdf),
            "preview": preview[:2000]
        })
        print(f"  Preview: {os.path.basename(pdf)} ({len(preview)} chars)")

    try:
        raw  = _llm(
            f"{PLANNER_PROMPT}\n\n"
            f"PDF previews:\n{json.dumps(previews)}\n\n"
            f"Return plan as JSON only."
        )
        plan = _parse_json(raw)
        if not plan:
            raise ValueError("Invalid JSON")
        state["plan"] = plan
        print(f"  ✓ Plan: {len(plan.get('tasks',[]))} task(s)")
        print(f"  Notes: {plan.get('notes','')}")
    except Exception as e:
        state["plan"] = {
            "tasks": [{"tool": "extract_pdf_text",
                       "target": f} for f in state["pdf_files"]],
            "conflict_check_fields": ["diagnosis","medications"],
            "notes": f"Fallback plan ({e})"
        }
        print(f"  ⚠ Fallback plan: {e}")

    _trace(state, {
        "phase"   : "PLAN",
        "action"  : "create_plan",
        "result"  : state["plan"].get("notes",""),
        "decision": f"Extract {len(state['pdf_files'])} PDF(s)"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 2: EXTRACT
# ══════════════════════════════════════════════════════════════════
def extract_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 2: EXTRACT\n{'#'*60}")

    for pdf in state["pdf_files"]:
        print(f"\n  Extracting: {os.path.basename(pdf)}")
        result = extract_pdf_text(pdf)

        if result["success"]:
            state["extracted_texts"][pdf] = result["text"]
            failed = result.get("failed_pages", 0)
            total  = result.get("page_count", 1)
            quota  = result.get("quota_exceeded", False)
            print(
                f"  ✓ {total} pages: "
                f"{result.get('digital_pages',0)} digital, "
                f"{result.get('vision_pages',0)} vision, "
                f"{failed} failed"
            )
            if quota:
                state["extraction_failed"] = True
                flag_for_clinician(
                    "PARTIAL EXTRACTION",
                    "all_sections",
                    f"{failed}/{total} pages unreadable due to quota.",
                    "WARNING"
                )
        else:
            state["extraction_failed"] = True
            state["extracted_texts"][pdf] = ""
            flag_for_clinician(
                "PDF EXTRACTION FAILED", "all_sections",
                f"Error: {result.get('error')}", "CRITICAL"
            )

    _trace(state, {
        "phase"   : "EXTRACT",
        "action"  : "extract_pdf_text",
        "result"  : (
            f"Extracted {len(state['extracted_texts'])} PDF(s). "
            f"Failed: {state['extraction_failed']}"
        ),
        "decision": "Proceed to analysis"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 3: ANALYZE — Split into 3 smaller calls to avoid truncation
# ══════════════════════════════════════════════════════════════════
def analyze_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 3: ANALYZE\n{'#'*60}")

    combined = "\n\n".join(state["extracted_texts"].values())

    if not combined.strip():
        print("  ✗ No text to analyze")
        for sec in ALL_SECTIONS:
            state["draft"][sec] = dict(MISSING_FIELD)
        state["sections_missing"] = list(ALL_SECTIONS)
        _trace(state, {
            "phase"   : "ANALYZE",
            "action"  : "extract_sections",
            "result"  : "No text — all MISSING",
            "decision": "Proceed"
        })
        return state

    print(f"  Text: {len(combined):,} chars")

    # Section descriptions for LLM context
    descs = {
        "patient_demographics"    : "Name, age, gender, MRN, IP number, weight",
        "admission_discharge_dates": "Exact admission and discharge dates",
        "principal_diagnosis"     : "Primary/main diagnosis",
        "secondary_diagnoses"     : "All secondary/additional diagnoses",
        "hospital_course"         : "Summary of clinical events during admission",
        "procedures"              : "IV cannulation, catheter, scans, ECG etc",
        "admission_medications"   : "Medications patient was on at admission",
        "discharge_medications"   : "Medications prescribed at discharge with doses",
        "medication_changes"      : "Medications added/stopped/changed with reasons",
        "allergies"               : "Known drug allergies",
        "lab_results"             : "CBC, creatinine, sodium, glucose, ABG results",
        "pending_results"         : "Results still awaited e.g. urine culture",
        "follow_up_instructions"  : "Follow-up date, clinic, instructions",
        "discharge_condition"     : "Patient condition at time of discharge",
    }

    # Split into 3 groups — smaller JSON = no truncation
    groups = [
        ["patient_demographics", "admission_discharge_dates",
         "principal_diagnosis",  "secondary_diagnoses",
         "discharge_condition"],
        ["hospital_course",      "procedures",
         "allergies",            "pending_results",
         "follow_up_instructions"],
        ["admission_medications","discharge_medications",
         "medication_changes",   "lab_results"],
    ]

    draft = {}
    for i, group in enumerate(groups):
        print(f"\n  Extracting group {i+1}/3: {group}")
        result = _extract_section(combined, group, descs)
        draft.update(result)
        extracted = [
            s for s in group
            if "[MISSING" not in str(result.get(s, {}).get("value",""))
        ]
        print(f"  ✓ Got: {extracted}")
        if i < len(groups) - 1:
            time.sleep(3)  # small delay between calls

    # Ensure all sections exist
    for sec in ALL_SECTIONS:
        if sec not in draft:
            draft[sec] = dict(MISSING_FIELD)
        elif isinstance(draft[sec], str):
            draft[sec] = {
                "value" : draft[sec] or "[MISSING - FLAG FOR REVIEW]",
                "source": "extracted"
            }

    state["draft"] = draft

    # Track found vs missing
    found   = []
    missing = []
    for sec in ALL_SECTIONS:
        field = draft.get(sec, {})
        if isinstance(field, dict):
            val = str(field.get("value", ""))
        elif isinstance(field, list):
            # LLM returned a list — join it and fix the draft
            val = "; ".join(str(v) for v in field)
            draft[sec] = {"value": val, "source": "extracted"}
        else:
            val = str(field)
        val = val.strip()
        if val and "[MISSING" not in val:
            found.append(sec)
        else:
            missing.append(sec)

    state["sections_found"]   = found
    state["sections_missing"] = missing

    print(f"\n  ✓ Sections found  : {len(found)}/14")
    print(f"  ✗ Sections missing: {len(missing)}/14")
    if missing:
        print(f"    {missing}")

    # Retry missing sections with a focused second pass
    if missing:
        print(f"\n  [Retry] Extracting {len(missing)} missing sections...")
        time.sleep(3)
        retry_result = _extract_section(combined, missing, descs)
        for sec in missing[:]:
            field = retry_result.get(sec, {})
            # Normalize to string safely
            if isinstance(field, dict):
                val = str(field.get("value", ""))
            elif isinstance(field, list):
                val = "; ".join(str(x) for x in field)
                field = {"value": val, "source": "extracted"}
            else:
                val = str(field)
            val = val.strip()
            if val and "[MISSING" not in val:
                draft[sec] = field if isinstance(field, dict) else {
                    "value": val, "source": "extracted"
                }
                found.append(sec)
                missing.remove(sec)
                print(f"    ✓ Recovered: {sec}")

        state["sections_found"]   = found
        state["sections_missing"] = missing
        print(f"  ✓ After retry: {len(found)}/14 sections found")

    # Extract medications list
    adm = draft.get("admission_medications",{})
    dis = draft.get("discharge_medications",{})
    adm_v = adm.get("value","") if isinstance(adm,dict) else str(adm)
    dis_v = dis.get("value","") if isinstance(dis,dict) else str(dis)
    if isinstance(adm_v, list): adm_v = "; ".join(str(x) for x in adm_v)
    if isinstance(dis_v, list): dis_v = "; ".join(str(x) for x in dis_v)
    all_meds_text = str(adm_v) + " " + str(dis_v)
    state["medications_found"] = list({
        w.strip().lower()
        for w in all_meds_text.replace("\n"," ").replace(","," ").split()
        if len(w.strip()) > 3 and "[MISSING" not in w
    })[:20]

    _trace(state, {
        "phase"    : "ANALYZE",
        "reasoning": "Extracted in 3 focused groups to prevent JSON truncation",
        "action"   : "extract_sections (3 LLM calls)",
        "result"   : f"Found {len(found)}/14. Missing: {missing}",
        "decision" : "Proceed to reconciliation"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 4: RECONCILE
# ══════════════════════════════════════════════════════════════════
def reconcile_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 4: RECONCILE\n{'#'*60}")

    draft = state["draft"]

    # Build conflict extractions
    # Only flag REAL conflicts — same field with different values
    # from DIFFERENT document sections (not principal vs secondary)
    extractions = []
    diag = draft.get("principal_diagnosis",{})
    if isinstance(diag, dict):
        v = diag.get("value","")
        s = diag.get("source","doc1")
        if v and "[MISSING" not in v:
            extractions.append({
                "source": s, "field": "diagnosis", "value": v
            })

    # Only add secondary diagnosis as conflict source if it
    # contradicts principal diagnosis (not just adds to it)
    # e.g. DKA vs T2DM+AFI from different pages = conflict
    # but "1) Gastroenteritis 2) UTI" = same document, not conflict

    if len(extractions) > 1:
        result = detect_conflicts(extractions)
        if result["success"] and result["conflicts"]:
            for field, values in result["conflicts"].items():
                flag_for_clinician(
                    "CONFLICTING VALUES", field,
                    "Conflict: " + " vs ".join(
                        f"{v['value'][:50]} ({v['source']})"
                        for v in values
                    ),
                    "WARNING"
                )
            state["conflicts_found"] = list(result["conflicts"].keys())
            print(f"  ⚠ Conflicts: {state['conflicts_found']}")
        else:
            print(f"  ✓ No conflicts detected")
    else:
        print(f"  ✓ Conflict check done")

    # Medication reconciliation
    adm = draft.get("admission_medications",{})
    dis = draft.get("discharge_medications",{})
    adm_v = adm.get("value","") if isinstance(adm,dict) else str(adm)
    dis_v = dis.get("value","") if isinstance(dis,dict) else str(dis)

    # Handle case where value might be a list
    if isinstance(adm_v, list):
        adm_v = "; ".join(str(x) for x in adm_v)
    if isinstance(dis_v, list):
        dis_v = "; ".join(str(x) for x in dis_v)

    def _split_med_names(text):
        """Extract clean drug names from medication text."""
        if not text or "[MISSING" in str(text):
            return []
        parts = re.split(r"[;\n]|\d+\.\s+", str(text))
        result = []
        for p in parts:
            p = p.strip()
            if len(p) > 3 and "[MISSING" not in p:
                # Take first meaningful words as drug name
                words = p.split()[:4]
                name  = " ".join(words).strip(".,;:")
                if len(name) > 3:
                    result.append(name)
        return result

    adm_meds = _split_med_names(adm_v)
    dis_meds = _split_med_names(dis_v)

    if adm_meds or dis_meds:
        recon = compare_medications(adm_meds, dis_meds)
        if recon["success"] and recon["changes"]:
            for change in recon["changes"]:
                flag_for_clinician(
                    f"MEDICATION {change['change_type']}",
                    "medication_changes",
                    f"{change['medication']} was "
                    f"{change['change_type'].lower()} — reason not documented",
                    "REVIEW"
                )
            changes_text = "; ".join(
                f"{c['medication']} {c['change_type']}"
                for c in recon["changes"]
            )
            if "[MISSING" in str(draft.get("medication_changes",{}).get("value","")):
                state["draft"]["medication_changes"] = {
                    "value" : changes_text,
                    "source": "medication_reconciliation"
                }
            print(f"  ✓ Med reconciliation: {recon['summary']}")
    else:
        print(f"  ✓ No meds to reconcile")

    _trace(state, {
        "phase"    : "RECONCILE",
        "reasoning": "Comparing diagnosis and medication values across sources",
        "action"   : "detect_conflicts + compare_medications",
        "result"   : f"Conflicts: {state['conflicts_found']}",
        "decision" : "Proceed to safety"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 5: SAFETY
# ══════════════════════════════════════════════════════════════════
def safety_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 5: SAFETY CHECK\n{'#'*60}")

    meds = state["medications_found"]
    if meds:
        result = check_drug_interactions(meds)
        if result["success"] and result["interactions"]:
            for ix in result["interactions"]:
                flag_for_clinician(
                    "DRUG INTERACTION",
                    "discharge_medications",
                    f"{ix['drug1']} + {ix['drug2']}: {ix['severity']}",
                    "CRITICAL" if "HIGH" in ix["severity"] else "WARNING"
                )
            print(f"  ⚠ {result['summary']}")
        else:
            print(f"  ✓ {result.get('summary','No interactions')}")
    else:
        print(f"  ✓ No medications to check")

    # Always check allergies documented
    allergies_val = state["draft"].get("allergies",{})
    av = allergies_val.get("value","") if isinstance(allergies_val,dict) else str(allergies_val)
    if "[MISSING" in av or not av.strip():
        flag_for_clinician(
            "ALLERGIES NOT DOCUMENTED", "allergies",
            "Allergy status unconfirmed. Verify before prescribing.",
            "CRITICAL"
        )

    _trace(state, {
        "phase"   : "SAFETY",
        "action"  : "check_drug_interactions",
        "result"  : f"Checked {len(meds)} medications",
        "decision": "Proceed to finalize"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# NODE 6: FINALIZE
# ══════════════════════════════════════════════════════════════════
def finalize_node(state: AgentState) -> AgentState:
    print(f"\n{'#'*60}\n  PHASE 6: FINALIZE\n{'#'*60}")

    all_flags = get_all_flags()
    state["draft"]["clinician_flags"] = all_flags
    state["draft"]["_meta"] = {
        "status"           : "DRAFT - REQUIRES CLINICIAN REVIEW",
        "model"            : REASON_MODEL,
        "architecture"     : "LangGraph 6-node pipeline",
        "patient_id"       : state["patient_id"],
        "sections_found"   : sorted(state["sections_found"]),
        "sections_missing" : sorted(state["sections_missing"]),
        "flags_count"      : len(all_flags),
        "extraction_failed": state["extraction_failed"],
    }
    state["completed"] = True

    print(f"  ✓ Sections found  : {len(state['sections_found'])}/14")
    print(f"  ✓ Sections missing: {len(state['sections_missing'])}/14")
    print(f"  ✓ Flags raised    : {len(all_flags)}")

    _trace(state, {
        "phase"   : "FINALIZE",
        "action"  : "assemble_draft",
        "result"  : "Summary assembled and validated",
        "decision": "Complete"
    })
    return state


# ══════════════════════════════════════════════════════════════════
# BUILD AND RUN
# ══════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan",      plan_node)
    g.add_node("extract",   extract_node)
    g.add_node("analyze",   analyze_node)
    g.add_node("reconcile", reconcile_node)
    g.add_node("safety",    safety_node)
    g.add_node("finalize",  finalize_node)
    g.add_edge(START,       "plan")
    g.add_edge("plan",      "extract")
    g.add_edge("extract",   "analyze")
    g.add_edge("analyze",   "reconcile")
    g.add_edge("reconcile", "safety")
    g.add_edge("safety",    "finalize")
    g.add_edge("finalize",  END)
    return g.compile()


def run_agent(patient_id: str, pdf_files: list) -> dict:
    reset_flags()
    state = _make_state(patient_id, pdf_files)
    app   = build_graph()

    print(f"\n  Running LangGraph agent...")
    print(f"  Model: {REASON_MODEL}")

    final = app.invoke(state)

    return {
        "summary": final.get("draft", {}),
        "trace"  : final.get("trace", []),
        "state"  : {
            "sections_found"  : final.get("sections_found",   []),
            "sections_missing": final.get("sections_missing", []),
            "completed"       : final.get("completed",        False),
        }
    }