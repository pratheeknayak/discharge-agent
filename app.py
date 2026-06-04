"""
app.py — Streamlit Web UI (Fixed display version)
"""

import streamlit as st
import os
import json
import tempfile
import shutil
from datetime import datetime

st.set_page_config(
    page_title="Discharge Summary Agent",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.flag-critical{background:#7f1d1d;border-left:5px solid #ef4444;
               padding:10px 14px;margin:5px 0;border-radius:6px;
               color:#fecaca !important;}
.flag-critical strong{color:#fca5a5 !important;}
.flag-critical small{color:#fca5a5 !important;}
.flag-warning {background:#713f12;border-left:5px solid #f59e0b;
               padding:10px 14px;margin:5px 0;border-radius:6px;
               color:#fde68a !important;}
.flag-warning strong{color:#fcd34d !important;}
.flag-warning small{color:#fde68a !important;}
.flag-review  {background:#1e3a5f;border-left:5px solid #3b82f6;
               padding:10px 14px;margin:5px 0;border-radius:6px;
               color:#bfdbfe !important;}
.flag-review strong{color:#93c5fd !important;}
.flag-review small{color:#bfdbfe !important;}
.med-table    {width:100%;border-collapse:collapse;font-size:0.9em;}
.med-table th {background:#1e3a5f;color:white;padding:8px;text-align:left;}
.med-table td {padding:7px 8px;border-bottom:1px solid #334155;}
.med-table tr:hover{background:#1e293b;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.title("🏥 Discharge Summary Agent")
st.markdown(
    "**LangGraph + Gemini 2.5 Flash** · "
    "Agentic AI for clinical discharge summaries"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key", type="password",
        value=os.getenv("GEMINI_API_KEY", "")
    )
    model = st.selectbox(
        "Model",
        ["models/gemini-2.5-flash",
         "models/gemini-2.0-flash",
         "models/gemini-2.0-flash-lite"],
        index=0
    )
    run_part2 = st.checkbox("Run Part 2 (Doctor Review)", value=False)

    st.divider()
    st.markdown("**6-Phase Architecture**")
    for phase in ["1. PLAN","2. EXTRACT (Vision)","3. ANALYZE",
                  "4. RECONCILE","5. SAFETY","6. FINALIZE"]:
        st.markdown(f"→ {phase}")
    st.divider()
    st.markdown("**Safety Guarantees**")
    for g in ["No hallucination","Evidence tracking",
              "Conflict detection","Drug interactions",
              "Pydantic validation"]:
        st.markdown(f"✅ {g}")

# ── Upload ────────────────────────────────────────────────────────
st.subheader("📄 Upload Patient PDFs")
uploaded_files = st.file_uploader(
    "Upload all clinical documents for one patient",
    type=["pdf"], accept_multiple_files=True
)
patient_id = st.text_input("Patient ID", value="patient_001")

if not api_key:
    st.warning("Enter your Gemini API key in the sidebar")
if not uploaded_files:
    st.info("Upload at least one patient PDF to begin")

run_btn = st.button(
    "🚀 Run Discharge Summary Agent",
    type="primary",
    disabled=not (uploaded_files and api_key)
)


# ── Helper: render any field value cleanly ───────────────────────
def render_field_value(field):
    """
    Render a section value regardless of its type.
    Handles: str, list, dict, nested structures.
    Returns (display_text, source)
    """
    if not field:
        return "[MISSING - FLAG FOR REVIEW]", "not_found"

    # EvidenceField format: {value, source}
    if isinstance(field, dict) and "value" in field:
        raw    = field["value"]
        source = field.get("source", "unknown")
        return _clean_value(raw), source

    # Plain string
    if isinstance(field, str):
        return _clean_value(field), "extracted"

    # Plain list
    if isinstance(field, list):
        text = _render_list(field)
        return text, "extracted"

    return str(field), "extracted"


def _clean_value(val):
    """Convert any value to clean display string."""
    if isinstance(val, list):
        return _render_list(val)
    if isinstance(val, dict):
        return _render_dict(val)
    return str(val) if val else "[MISSING - FLAG FOR REVIEW]"


def _render_list(items):
    """Convert list to numbered display."""
    if not items:
        return "[MISSING - FLAG FOR REVIEW]"
    cleaned = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            cleaned.append(f"{i}. {_render_dict(item)}")
        else:
            cleaned.append(f"{i}. {item}")
    return "\n".join(cleaned)


def _render_dict(d):
    """Convert dict to readable key: value string."""
    if not d:
        return ""
    parts = []
    for k, v in d.items():
        if k in ("value", "source"):
            continue
        if v and str(v) not in ("[MISSING - FLAG FOR REVIEW]", ""):
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else str(d)


def render_medications_table(field):
    """
    Render medications as a clean HTML table.
    Handles both structured list and plain text.
    """
    if not field:
        return None, "not_found"

    value  = field.get("value",  field) if isinstance(field, dict) else field
    source = field.get("source", "extracted") if isinstance(field, dict) else "extracted"

    # If it's a list of medication dicts
    if isinstance(value, list) and value:
        rows = []
        for item in value:
            if isinstance(item, dict):
                name  = item.get("name",      item.get("drug",""))
                dose  = item.get("dosage",    item.get("dose",""))
                freq  = item.get("frequency", "")
                dur   = item.get("duration",  "")
                route = item.get("route",     "")
                rows.append(f"""
                <tr>
                  <td>{name}</td>
                  <td>{dose}</td>
                  <td>{route}</td>
                  <td>{freq}</td>
                  <td>{dur}</td>
                </tr>""")
            else:
                rows.append(
                    f'<tr><td colspan="5">{item}</td></tr>'
                )
        if rows:
            table_html = f"""
<table class="med-table">
  <thead><tr>
    <th>Medication</th><th>Dose</th>
    <th>Route</th><th>Frequency</th><th>Duration</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""
            return table_html, source

    # Plain string
    text = str(value) if value else "[MISSING - FLAG FOR REVIEW]"
    return None, source, text


# ── Main run ──────────────────────────────────────────────────────
if run_btn and uploaded_files and api_key:
    os.environ["GEMINI_API_KEY"]      = api_key
    os.environ["GEMINI_VISION_MODEL"] = model
    os.environ["GEMINI_REASON_MODEL"] = model

    tmp_dir   = tempfile.mkdtemp()
    pdf_paths = []
    for uf in uploaded_files:
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.read())
        pdf_paths.append(path)

    st.divider()
    st.subheader("🔄 Running Agent...")

    try:
        from agent.graph import run_agent
        from tools.clinical_tools import reset_flags
        reset_flags()

        with st.spinner(
            "⏳ Running 6-phase agent (3-5 minutes)... "
            "Reading PDFs with Gemini Vision"
        ):
            result = run_agent(patient_id, pdf_paths)

        summary = result.get("summary", {})
        trace   = result.get("trace",   [])
        meta    = summary.get("_meta",  {})
        flags   = summary.get("clinician_flags", [])
        found   = meta.get("sections_found",   [])
        missing = meta.get("sections_missing", [])

        shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── Metrics ───────────────────────────────────────────────
        st.divider()
        st.subheader("✅ Agent Complete")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sections Found",   f"{len(found)}/14",
                  delta=f"{len(found)-7} vs baseline")
        c2.metric("Sections Missing", len(missing),
                  delta=f"-{14-len(missing)} recovered")
        c3.metric("Clinician Flags",  len(flags))
        c4.metric("Model", model.split("/")[-1])

        # ── Flags ─────────────────────────────────────────────────
        if flags:
            st.subheader(f"⚠️ {len(flags)} Clinician Flag(s)")
            for flag in flags:
                sev     = flag.get("severity","REVIEW").upper()
                section = flag.get("section","?").upper()
                reason  = flag.get("reason", "?")
                details = flag.get("details","")
                icon    = {"CRITICAL":"🔴","WARNING":"🟡","REVIEW":"🔵"}.get(sev,"🔵")
                label   = f"{icon} **[{sev}] [{section}]** — {reason}"
                if sev == "CRITICAL":
                    st.error(f"{label}\n\n{details}")
                elif sev == "WARNING":
                    st.warning(f"{label}\n\n{details}")
                else:
                    st.info(f"{label}\n\n{details}")

        # ── Summary Sections ──────────────────────────────────────
        st.subheader("📋 Discharge Summary Draft")
        st.caption(
            "⚠️ DRAFT ONLY — requires clinician review before use"
        )

        ALL_SECTIONS = [
            ("patient_demographics",
             "👤 Patient Demographics",      False),
            ("admission_discharge_dates",
             "📅 Admission & Discharge Dates",False),
            ("principal_diagnosis",
             "🔴 Principal Diagnosis",        False),
            ("secondary_diagnoses",
             "🟡 Secondary Diagnoses",        False),
            ("hospital_course",
             "📊 Hospital Course",            False),
            ("procedures",
             "🔧 Procedures",                 False),
            ("admission_medications",
             "💊 Admission Medications",      True),
            ("discharge_medications",
             "💊 Discharge Medications",      True),
            ("medication_changes",
             "⚠️ Medication Changes",         False),
            ("allergies",
             "🚨 Allergies",                  False),
            ("lab_results",
             "🧪 Lab Results",                False),
            ("pending_results",
             "⏳ Pending Results",            False),
            ("follow_up_instructions",
             "📌 Follow-up Instructions",     False),
            ("discharge_condition",
             "🏥 Discharge Condition",        False),
        ]

        for sec_key, sec_label, is_med in ALL_SECTIONS:
            field      = summary.get(sec_key, {})
            is_missing = False

            # Determine if missing
            if isinstance(field, dict):
                val = str(field.get("value",""))
            elif isinstance(field, list):
                val = str(field)
            else:
                val = str(field)
            is_missing = "[MISSING" in val or not val.strip()

            icon = "❌" if is_missing else "✅"

            with st.expander(
                f"{sec_label} {icon}",
                expanded=not is_missing
            ):
                if is_missing:
                    st.error("[MISSING - FLAG FOR REVIEW]")
                    continue

                # Medication sections — try table format
                if is_med:
                    med_result = render_medications_table(field)
                    if len(med_result) == 2:
                        table_html, source = med_result
                        st.markdown(table_html,
                                    unsafe_allow_html=True)
                        st.caption(f"Source: {source}")
                    else:
                        table_html, source, plain = med_result
                        if table_html:
                            st.markdown(table_html,
                                        unsafe_allow_html=True)
                        else:
                            st.write(plain)
                        st.caption(f"Source: {source}")
                    continue

                # All other sections
                display, source = render_field_value(field)
                st.write(display)
                st.caption(f"Source: {source}")

        # ── Agent Trace ───────────────────────────────────────────
        st.subheader("🔍 Agent Trace")
        st.caption("Step-by-step reasoning of the 6-phase agent")

        phase_icons = {
            "PLAN"     : "📋",
            "EXTRACT"  : "📄",
            "ANALYZE"  : "🔬",
            "RECONCILE": "⚖️",
            "SAFETY"   : "🛡️",
            "FINALIZE" : "✅",
        }

        for step in trace:
            phase    = step.get("phase",    "")
            action   = step.get("action",   "")
            result_v = str(step.get("result",""))[:300]
            reasoning= step.get("reasoning","")
            decision = step.get("decision", "")
            icon     = phase_icons.get(phase, "▶️")

            with st.expander(
                f"{icon} Phase {phase} — {action}",
                expanded=False
            ):
                cols = st.columns([1,2])
                with cols[0]:
                    st.markdown("**Phase**")
                    st.code(phase)
                    st.markdown("**Action**")
                    st.code(action)
                with cols[1]:
                    if reasoning:
                        st.markdown(f"**Reasoning:** {reasoning}")
                    st.markdown(f"**Result:** {result_v}")
                    if decision:
                        st.markdown(f"**Decision:** {decision}")

        # ── Downloads ─────────────────────────────────────────────
        st.subheader("💾 Download Results")
        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "📥 Summary JSON",
                data=json.dumps(summary, indent=2),
                file_name=f"{patient_id}_summary.json",
                mime="application/json",
                use_container_width=True
            )
        with c2:
            trace_lines = []
            for s in trace:
                trace_lines += [
                    f"PHASE    : {s.get('phase','')}",
                    f"ACTION   : {s.get('action','')}",
                    f"REASONING: {s.get('reasoning','')}",
                    f"RESULT   : {s.get('result','')}",
                    f"DECISION : {s.get('decision','')}",
                    "─" * 50
                ]
            st.download_button(
                "📥 Agent Trace TXT",
                data="\n".join(trace_lines),
                file_name=f"{patient_id}_trace.txt",
                mime="text/plain",
                use_container_width=True
            )

        # ── Part 2 ────────────────────────────────────────────────
        if run_part2:
            st.divider()
            st.subheader("🎓 Part 2: Simulated Doctor Review")
            with st.spinner("Applying doctor editing policy..."):
                try:
                    from part2.reviewer import simulate_doctor_review
                    from part2.metrics  import (
                        compute_edit_distance,
                        compute_section_scores
                    )
                    review = simulate_doctor_review(summary)
                    edited = review["edited"]
                    ed     = compute_edit_distance(summary, edited)
                    scores = compute_section_scores(summary, edited)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Edit Distance", f"{ed:.4f}",
                              help="0=identical, 1=completely different")
                    best  = max(scores, key=scores.get, default="N/A")
                    worst = min(scores, key=scores.get, default="N/A")
                    c2.metric("Best Section",
                              f"{best} ({scores.get(best,0):.2f})")
                    c3.metric("Worst Section",
                              f"{worst} ({scores.get(worst,0):.2f})")

                    st.download_button(
                        "📥 Download Doctor-Edited Version",
                        data=json.dumps(edited, indent=2),
                        file_name=f"{patient_id}_edited.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Part 2 error: {e}")

    except Exception as e:
        import traceback
        st.error(f"Agent error: {e}")
        st.code(traceback.format_exc())
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🏥 Discharge Summary Agent · "
    "LangGraph + Gemini 2.5 Flash · "
    "All outputs are DRAFTS for clinician review only"
)