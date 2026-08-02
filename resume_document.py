"""
Multimodal resume ingestion: turns a PDF resume into clean text before it
reaches the scoring pipeline.

Primary path: Gemini's native document understanding — the PDF is sent
directly to the already-configured `llm` client from nodes.py as a document
part, no separate OCR step needed for a normal digital PDF.

Fallback path: local OCR (pdf2image + pytesseract), used only if the Gemini
call fails after retries. This mirrors the project's existing
Gemini -> Groq fallback shape, but Groq has no document-input support, so a
local OCR fallback stands in for it here instead.

New dependencies (not yet in requirements.txt):
    pip install pdf2image pytesseract
    + poppler-utils and tesseract-ocr installed at the OS level
These are only imported inside `_ocr_fallback`, so this module still loads
fine (and the Gemini path still works) even if they're not installed.
"""
import base64
import time

from langchain_core.messages import HumanMessage

from nodes import llm  # reuse the already-configured Gemini client, don't reconfigure


def _gemini_document_understanding(file_path: str, max_retries: int = 2) -> str:
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract all text from this resume PDF. Preserve section "
                    "headers (EDUCATION, EXPERIENCE, SKILLS, PROJECTS, etc.) "
                    "in uppercase on their own line, matching how a plain-text "
                    "resume would be laid out. Return only the extracted text, "
                    "no commentary."
                ),
            },
            {
                "type": "media",
                "mime_type": "application/pdf",
                "data": b64_pdf,
            },
        ]
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke([message])
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(
                f"\n[DEBUG] Gemini document understanding failed "
                f"(attempt {attempt + 1}): {e}. Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"Gemini document understanding exhausted retries: {last_err}")


def _ocr_fallback(file_path: str) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    pages = convert_from_path(file_path)
    text_chunks = [pytesseract.image_to_string(page) for page in pages]
    return "\n".join(text_chunks).strip()


def parse_resume_document(file_path: str) -> dict:
    """Returns a plain dict matching models.ParseResumeDocumentOutput's
    shape — kept as a dict here so this module has no Pydantic dependency of
    its own; mcp_server.py wraps it into the typed model at the tool
    boundary."""
    try:
        text = _gemini_document_understanding(file_path)
        if text and text.strip():
            return {
                "extracted_text": text,
                "source": "gemini_document_understanding",
                "status": "ok",
                "error": None,
            }
        raise RuntimeError("Gemini returned empty text")
    except Exception as e:
        print(f"\n[DEBUG] Falling back to OCR for {file_path}: {e}")
        try:
            text = _ocr_fallback(file_path)
            if not text:
                return {
                    "extracted_text": None,
                    "source": "ocr_fallback",
                    "status": "generation_failed",
                    "error": "OCR produced no text",
                }
            return {
                "extracted_text": text,
                "source": "ocr_fallback",
                "status": "ok",
                "error": None,
            }
        except Exception as ocr_e:
            return {
                "extracted_text": None,
                "source": "none",
                "status": "generation_failed",
                "error": f"Gemini failed ({e}); OCR fallback also failed ({ocr_e})",
            }
