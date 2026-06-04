"""
tools/pdf_extractor.py — Final version
========================================
Fixes:
1. None response check → batch retried once
2. Failed batch retry with smaller zoom for large pages
3. Better error messages
"""

import fitz
import os
import io
import time
import PIL.Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client      = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "models/gemini-2.5-flash")
BATCH_SIZE   = 20
BATCH_DELAY  = 3


def _call_vision(contents, retries=3, wait=15):
    """Call vision API with retry on rate limits and None response."""
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(
                model=VISION_MODEL,
                contents=contents
            )
            # Fix 1: check for None response
            if resp is None or resp.text is None:
                raise ValueError("API returned None response")
            return resp.text
        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err:
                w = wait * (attempt + 1)
                print(f"    [Rate limit] waiting {w}s "
                      f"(attempt {attempt+1}/{retries})...")
                time.sleep(w)
            elif "None response" in err and attempt < retries - 1:
                print(f"    [Retry] None response, retrying...")
                time.sleep(5)
            else:
                raise
    raise Exception(f"Vision failed after {retries} retries")


def _ocr_fallback(page) -> str:
    """pytesseract OCR fallback."""
    try:
        import pytesseract
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def _process_batch(doc, batch: list, zoom=1.5) -> dict:
    """
    Process a batch of pages with Gemini Vision.
    Returns {page_idx: text} dict.
    """
    nums = [p + 1 for p in batch]

    prompt = (
        f"These are {len(batch)} pages from a hospital patient record "
        f"(pages {nums}).\n"
        f"Transcribe ALL visible text for EACH page.\n"
        f"Format EXACTLY as:\n"
        f"=== PAGE N ===\n<transcription>\n\n"
        f"where N is the actual page number.\n"
        f"Mark unreadable text as [ILLEGIBLE].\n"
        f"No commentary — transcription only."
    )

    contents = [prompt]
    for idx in batch:
        mat       = fitz.Matrix(zoom, zoom)
        pix       = doc[idx].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img       = PIL.Image.open(io.BytesIO(img_bytes))
        contents.append(img)

    batch_text = _call_vision(contents)

    # Parse === PAGE N === markers
    result = {}
    for idx in batch:
        pnum   = idx + 1
        marker = f"=== PAGE {pnum} ==="
        if marker in batch_text:
            start = batch_text.index(marker) + len(marker)
            nxt   = [
                batch_text.index(f"=== PAGE {p+1} ===")
                for p in batch
                if p != idx
                and f"=== PAGE {p+1} ===" in batch_text
                and batch_text.index(f"=== PAGE {p+1} ===") > start
            ]
            end         = min(nxt) if nxt else len(batch_text)
            result[idx] = batch_text[start:end].strip()
        else:
            result[idx] = batch_text  # fallback: whole response

    return result


def extract_pdf_text(pdf_path: str) -> dict:
    """
    Extract all PDF pages with Gemini Vision.
    Retries failed batches once with smaller zoom.
    Falls back to pytesseract OCR if quota exhausted.

    Returns:
        {success, text, page_count, digital_pages,
         vision_pages, ocr_pages, failed_pages,
         quota_exceeded, error}
    """
    if not os.path.exists(pdf_path):
        return {
            "success": False, "text": "",
            "page_count": 0,
            "error": f"Not found: {pdf_path}"
        }

    try:
        doc        = fitz.open(pdf_path)
        page_count = len(doc)
        all_text   = [""] * page_count

        # Pass 1: PyMuPDF digital text
        handwritten   = []
        digital_count = 0
        for i in range(page_count):
            text = doc[i].get_text().strip()
            if len(text) >= 10:
                all_text[i] = f"--- Page {i+1} ---\n{text}"
                digital_count += 1
            else:
                handwritten.append(i)

        batches = [
            handwritten[i: i + BATCH_SIZE]
            for i in range(0, len(handwritten), BATCH_SIZE)
        ]

        print(f"    Digital pages  : {digital_count}")
        print(f"    Vision pages   : {len(handwritten)} "
              f"→ {len(batches)} batch(es) of {BATCH_SIZE}")
        print(f"    Vision model   : {VISION_MODEL}")

        # Pass 2: Vision batches
        quota_exhausted = False
        vision_count    = 0
        ocr_count       = 0
        failed_indices  = []

        for b_idx, batch in enumerate(batches):
            if quota_exhausted:
                failed_indices.extend(batch)
                continue

            nums = [p + 1 for p in batch]
            print(f"\n    [Vision] Batch {b_idx+1}/{len(batches)} "
                  f"pages {nums[0]}–{nums[-1]}...")

            try:
                page_texts = _process_batch(doc, batch, zoom=1.5)
                for idx, text in page_texts.items():
                    all_text[idx] = (
                        f"--- Page {idx+1} (handwritten) ---\n{text}"
                    )
                    vision_count += 1
                print(f"    ✓ Batch {b_idx+1} done ({len(batch)} pages)")

            except Exception as ve:
                err_str = str(ve)
                is_quota = (
                    "quota" in err_str.lower()
                    or "429" in err_str
                    or "exhausted" in err_str.lower()
                )

                if is_quota:
                    print(f"    ✗ Quota exhausted → OCR fallback")
                    quota_exhausted = True
                    failed_indices.extend(batch)
                else:
                    # Fix 2: retry failed batch with smaller zoom
                    print(f"    ✗ Batch {b_idx+1} error: {err_str[:60]}")
                    print(f"    → Retrying with smaller zoom...")
                    time.sleep(5)
                    try:
                        page_texts = _process_batch(
                            doc, batch, zoom=1.0
                        )
                        for idx, text in page_texts.items():
                            all_text[idx] = (
                                f"--- Page {idx+1} (handwritten) ---\n"
                                f"{text}"
                            )
                            vision_count += 1
                        print(f"    ✓ Retry succeeded")
                    except Exception as ve2:
                        print(f"    ✗ Retry failed: {str(ve2)[:60]}")
                        failed_indices.extend(batch)

            if b_idx < len(batches) - 1 and not quota_exhausted:
                time.sleep(BATCH_DELAY)

        # Pass 3: OCR fallback for failed pages
        if failed_indices:
            print(f"\n    [OCR] Trying {len(failed_indices)} failed pages...")
            for idx in failed_indices:
                ocr_text = _ocr_fallback(doc[idx])
                if ocr_text:
                    all_text[idx] = (
                        f"--- Page {idx+1} (OCR) ---\n{ocr_text}"
                    )
                    ocr_count += 1
                else:
                    all_text[idx] = (
                        f"--- Page {idx+1} ---\n"
                        f"[{'VISION QUOTA EXCEEDED' if quota_exhausted else 'VISION FAILED'}"
                        f" - handwritten page unreadable]"
                    )

        doc.close()

        # Fill empty slots
        for i in range(page_count):
            if not all_text[i]:
                all_text[i] = f"--- Page {i+1} ---\n[EMPTY]"

        final_text   = "\n\n".join(all_text)
        failed_pages = sum(
            1 for t in all_text
            if "[VISION" in t or "[EMPTY" in t
        )

        print(f"\n    ── Extraction complete ──────────────────")
        print(f"    Digital : {digital_count}")
        print(f"    Vision  : {vision_count}")
        print(f"    OCR     : {ocr_count}")
        print(f"    Failed  : {failed_pages}")

        return {
            "success"       : True,
            "text"          : final_text,
            "page_count"    : page_count,
            "digital_pages" : digital_count,
            "vision_pages"  : vision_count,
            "ocr_pages"     : ocr_count,
            "failed_pages"  : failed_pages,
            "quota_exceeded": quota_exhausted,
            "error"         : None
        }

    except Exception as e:
        return {
            "success": False, "text": "",
            "page_count": 0, "error": str(e),
            "digital_pages": 0, "vision_pages": 0,
            "ocr_pages": 0, "failed_pages": 0,
            "quota_exceeded": False
        }