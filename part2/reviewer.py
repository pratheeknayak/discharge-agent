"""
part2/reviewer.py
==================
Simulated Doctor Reviewer for Part 2 learning loop.
Applies consistent hidden editing policy to agent drafts.
Produces (draft, edited) pairs for measuring improvement.
"""

import os
import json
import copy
from google import genai
from dotenv import load_dotenv
from prompts.system import DOCTOR_REVIEW_PROMPT

load_dotenv()
_client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
REASON_MODEL = os.getenv("GEMINI_REASON_MODEL", "models/gemini-2.5-flash")


def simulate_doctor_review(draft: dict) -> dict:
    """
    Apply simulated doctor editing policy to a draft.

    Args:
        draft: agent's output summary dict

    Returns:
        { success: bool, edited: dict, error: str|None }
    """
    try:
        prompt = (
            f"{DOCTOR_REVIEW_PROMPT}\n\n"
            f"Apply editing policy to this draft.\n"
            f"Return corrected version as JSON only.\n\n"
            f"DRAFT:\n{json.dumps(draft, indent=2)}"
        )

        resp = _client.models.generate_content(
            model=REASON_MODEL,
            contents=prompt
        )

        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])

        edited = json.loads(raw)
        return {"success": True, "edited": edited, "error": None}

    except Exception as e:
        fallback = copy.deepcopy(draft)
        fallback["_review_error"] = f"Review failed: {e}"
        return {"success": False, "edited": fallback, "error": str(e)}