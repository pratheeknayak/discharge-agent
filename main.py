"""
main.py
========
Entry point for the Discharge Summary Agent.

Usage:
  python main.py                    # Part 1 only
  python main.py --part2            # Part 1 + Part 2
  python main.py --patients-dir <p> # custom folder
"""

import os
import sys
import json
import argparse
from datetime import datetime

from agent.graph import run_agent
from tools.clinical_tools import reset_flags


def save_outputs(patient_id: str, result: dict):
    """Save summary JSON and trace to outputs folder."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("traces",  exist_ok=True)

    # Summary JSON
    summary_path = f"outputs/{patient_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result["summary"], f, indent=2)

    # Human-readable trace
    trace_path = f"traces/{patient_id}_trace.txt"
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(f"DISCHARGE SUMMARY AGENT — STEP TRACE\n")
        f.write(f"Patient  : {patient_id}\n")
        f.write(f"Generated: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 65 + "\n")

        for step in result.get("trace", []):
            f.write(f"\n{'─'*55}\n")
            f.write(
                f"ITERATION {step.get('iteration','?')}  |  "
                f"PHASE: {step.get('phase', 'N/A')}\n"
            )
            f.write(f"{'─'*55}\n")
            f.write(f"REASONING:\n{step.get('reasoning','')}\n\n")
            f.write(f"ACTION : {step.get('action','')}\n")
            f.write(f"RESULT : {step.get('result','')}\n")
            f.write(f"DECISION: {step.get('decision','')}\n")

        f.write("\n" + "=" * 65 + "\n")
        f.write("END OF TRACE\n")

    print(f"\n  ✓ Summary → {summary_path}")
    print(f"  ✓ Trace   → {trace_path}")

    # Print flags
    flags = result["summary"].get("clinician_flags", [])
    if flags:
        print(f"\n  ⚠️  {len(flags)} CLINICIAN FLAG(S):")
        print(f"  {'─'*50}")
        for flag in flags:
            sev    = flag.get("severity", "REVIEW")
            sec    = flag.get("section",  "?").upper()
            reason = flag.get("reason",   "?")
            detail = flag.get("details",  "")[:80]
            print(f"  [{sev}] [{sec}] {reason}")
            print(f"    → {detail}")
    else:
        print("\n  ✓ No clinician flags raised")


def process_patient(
    patient_folder: str,
    memory_text: str = ""
) -> dict:
    """Run agent on one patient folder."""
    patient_id = os.path.basename(patient_folder)

    print(f"\n{'#'*65}")
    print(f"  PATIENT : {patient_id}")
    print(f"  FOLDER  : {patient_folder}")
    print(f"{'#'*65}")

    # Discover PDFs
    pdf_files = sorted([
        os.path.join(patient_folder, f)
        for f in os.listdir(patient_folder)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"  ERROR: No PDFs found in {patient_folder}")
        return {"summary": {}, "trace": []}

    print(f"  PDFs: {len(pdf_files)}")
    for p in pdf_files:
        print(f"    {os.path.basename(p)}")

    reset_flags()
    result = run_agent(patient_id, pdf_files)
    save_outputs(patient_id, result)
    return result


def run_part2(patient_id: str, summary: dict) -> dict:
    """Simulate doctor review and compute metrics."""
    from part2.reviewer import simulate_doctor_review
    from part2.memory   import store_correction, get_all_metrics
    from part2.metrics  import (
        compute_edit_distance,
        compute_section_scores,
        generate_improvement_report
    )

    print(f"\n  ── PART 2: DOCTOR REVIEW ─────────────────────")

    review = simulate_doctor_review(summary)
    edited = review["edited"]

    ed     = compute_edit_distance(summary, edited)
    scores = compute_section_scores(summary, edited)

    print(f"  Edit distance : {ed:.4f} (lower = better)")
    best  = max(scores, key=scores.get, default="N/A")
    worst = min(scores, key=scores.get, default="N/A")
    print(f"  Best section  : {best} ({scores.get(best, 0):.2f})")
    print(f"  Worst section : {worst} ({scores.get(worst, 0):.2f})")

    # Save edited version
    os.makedirs("outputs", exist_ok=True)
    edited_path = f"outputs/{patient_id}_edited.json"
    with open(edited_path, "w") as f:
        json.dump(edited, f, indent=2)
    print(f"  ✓ Edited → {edited_path}")

    store_correction(patient_id, summary, edited, ed, scores)

    return {
        "edited"        : edited,
        "edit_distance" : ed,
        "section_scores": scores
    }


def save_improvement_report():
    """Generate Part 2 improvement report."""
    from part2.memory  import get_all_metrics
    from part2.metrics import generate_improvement_report

    metrics = get_all_metrics()
    report  = generate_improvement_report(metrics)

    path = "outputs/improvement_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*65}")
    print("  PART 2 — IMPROVEMENT REPORT")
    print(f"{'═'*65}")

    if "error" in report:
        print(f"  {report['error']}")
        return

    print(f"  Patients    : {len(report['curve'])}")
    print(f"  Avg edit    : {report['avg_edit_distance']}")
    print(f"  Improvement : {report['total_improvement']}")
    print(f"  Best section: {report['best_section']}")
    print(f"  Worst section: {report['worst_section']}")
    print(f"\n  Curve:")
    for pt in report["curve"]:
        bar = "█" * int((1 - pt["edit_distance"]) * 30)
        print(f"    {pt['patient_id']:<20} "
              f"ed={pt['edit_distance']:.4f}  {bar}")
    print(f"\n  ✓ Report → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent"
    )
    parser.add_argument(
        "--part2", action="store_true",
        help="Run Part 2 learning loop"
    )
    parser.add_argument(
        "--patients-dir", default="patients",
        help="Path to patients folder"
    )
    args = parser.parse_args()

    patient_dir = args.patients_dir

    if not os.path.exists(patient_dir):
        print(f"ERROR: '{patient_dir}' not found")
        sys.exit(1)

    folders = sorted([
        os.path.join(patient_dir, d)
        for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d))
    ])

    if not folders:
        print(f"ERROR: No patient folders in '{patient_dir}'")
        sys.exit(1)

    print(f"\n{'═'*65}")
    print(f"  DISCHARGE SUMMARY AGENT")
    print(f"  Architecture    : LangGraph + Gemini 2.5 Flash")
    print(f"  Patients        : {len(folders)}")
    print(f"  Part 2 enabled  : {args.part2}")
    print(f"  Started         : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*65}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("traces",  exist_ok=True)

    for folder in folders:
        patient_id = os.path.basename(folder)

        # Get memory injection for Part 2
        memory_text = ""
        if args.part2:
            from part2.memory import build_memory_injection
            memory_text = build_memory_injection(max_examples=3)
            if memory_text:
                print(f"\n  [Memory] Injecting past corrections")

        # Run agent
        result = process_patient(folder, memory_text)

        # Part 2
        if args.part2 and result.get("summary"):
            run_part2(patient_id, result["summary"])

    # Part 2 report
    if args.part2:
        save_improvement_report()

    print(f"\n{'═'*65}")
    print("  ALL PATIENTS PROCESSED")
    print(f"  Outputs → outputs/")
    print(f"  Traces  → traces/")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()