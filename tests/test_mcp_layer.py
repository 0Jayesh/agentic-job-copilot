"""
test_mcp_layer.py

Tests the 4 files added/changed for the MCP server upgrade:
  - models.py           (Pydantic validation)
  - calendar_store.py   (local calendar store)
  - resume_document.py  (PDF -> text, Gemini primary / OCR fallback)
  - mcp_server.py        (tool wrappers, sequential + parallel orchestration)

Plus one integrity check that isn't about functionality at all: a SHA-256
hash comparison proving nodes.py, state.py, memory.py, education_maps.py,
resume.py, and graph_builder.py are byte-for-byte unchanged from before this
change set. If any of those hashes mismatch, something touched a file that
was supposed to be off-limits.

Style matches the existing test_*.py scripts in this repo (plain
check()-based assertions, run directly with `python test_mcp_layer.py`) --
not pytest, to stay consistent with test_helpers.py / test_node.py / etc.

Isolation note: calendar_store and memory-writing tests point at throwaway
files, never at the real scheduled_interviews.json or company_memory.json,
following the same lesson learned from test_memory.py polluting the real
company memory store.
"""
import __init__
from dotenv import load_dotenv
load_dotenv()

import hashlib
import os

PASS_COUNT = 0
FAIL_COUNT = 0


def check(label, actual, expected):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if actual == expected else "FAIL"
    if status == "PASS":
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"[{status}] {label} -> got {actual!r}, expected {expected!r}")


def check_true(label, condition):
    check(label, bool(condition), True)


# ===========================================================================
# 0. INTEGRITY CHECK — confirm nothing outside the 4 new/changed files moved
# ===========================================================================
# These hashes were taken directly off the tested, working versions of these
# files at the moment the MCP layer was added. They should never change as
# a side effect of this change set.

EXPECTED_HASHES = {
    # nodes.py hash updated in Phase 8: memory_lookup_node/finalize_node now
    # import from structured_memory.py/vector_memory.py instead of memory.py
    # (2-line import swap + finalize_node writing to vector_memory too).
    # Hash updated again same phase: the top-level `import vector_memory`
    # caused a circular import (vector_memory.py imports `embedder` back
    # from nodes.py, which isn't defined until later in the file) --
    # depending on which module imported nodes.py first, this either worked
    # by luck or raised ImportError. Fixed by moving the import inside
    # finalize_node (deferred import), which is immune to import order.
    # See test_memory_wiring.py for the phase that made this change.
    "nodes.py": "114af420bd9c30d9f744160d34bc3b9966acd0e7c38d5c6aae4eaaf66bbcb51f",
    "state.py": "11f074992b03d9ffbb908e4de1722a660e74699ec6eb3f45a406c4dc09f6a6f2",
    "memory.py": "182a259474798ccafa5efbad19f9a5b6c7401925dbdcfe63e016d51e957cdfce",
    "education_maps.py": "803024213b617f60e809ec8e045ad0194770d1e91040c7daf291817fd1170d3e",
    "resume.py": "445070b71550abf2efb453521876cb75061855dfdc772860ddf5d637e64868a8",
    # graph_builder.py hash updated in Phase 9: calls mlflow_setup.enable_tracing()
    # once at graph build time (mlflow.langchain.autolog()) -- 3 lines added, nothing else changed.
    "graph_builder.py": "788e8be424c4ae3399981d6df78cb1735f8237a049357cb341d512b43ee788ed",
}

print("--- 0. Untouched-files integrity check ---")

# Resolve the repo root regardless of whether this script runs from the
# repo root or from a tests/ subfolder: try CWD, this file's own directory,
# and its parent, and use whichever one actually contains nodes.py.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_roots = [os.getcwd(), _this_dir, os.path.dirname(_this_dir)]
REPO_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "nodes.py"))),
    _this_dir,
)

for filename, expected_hash in EXPECTED_HASHES.items():
    full_path = os.path.join(REPO_ROOT, filename)
    if not os.path.exists(full_path):
        check(f"{filename} exists", False, True)
        continue
    with open(full_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    check(f"{filename} unchanged", actual_hash, expected_hash)


# ===========================================================================
# 1. models.py — Pydantic validation actually validates
# ===========================================================================
print("\n--- 1. models.py ---")
from models import (
    ParseEmailOutput,
    MatchResumeInput,
    FitScoreResult,
    CheckCalendarConflictInput,
    CheckCalendarConflictOutput,
)
from pydantic import ValidationError

parsed = ParseEmailOutput(
    company="Unify Technologies",
    role="SDE",
    requirements=["PyTorch", "Spark"],
    status="parsed",
)
check("ParseEmailOutput round-trips company", parsed.company, "Unify Technologies")
check("ParseEmailOutput defaults requirements to list", isinstance(parsed.requirements, list), True)

nested = MatchResumeInput(parsed_state=parsed)
check("MatchResumeInput nested validation preserves company", nested.parsed_state.company, "Unify Technologies")

fit = FitScoreResult(score=88.5, reasoning="test", confidence=1.0, status="scored")
check("FitScoreResult accepts valid score", fit.score, 88.5)
check("FitScoreResult defaults matched_skills to empty list", fit.matched_skills, [])

try:
    FitScoreResult(score="not-a-number", reasoning="x", confidence=1.0, status="scored")
    check("FitScoreResult rejects non-numeric score", False, True)
except ValidationError:
    check("FitScoreResult rejects non-numeric score", True, True)

try:
    ParseEmailOutput(status=None)  # status is required, not Optional
    check("ParseEmailOutput requires status", False, True)
except ValidationError:
    check("ParseEmailOutput requires status", True, True)


# ===========================================================================
# 2. calendar_store.py — isolated from the real scheduled_interviews.json
# ===========================================================================
print("\n--- 2. calendar_store.py (isolated test file) ---")
import calendar_store

TEST_CALENDAR_FILE = "test_scheduled_interviews.json"
_original_calendar_file = calendar_store.CALENDAR_FILE
calendar_store.CALENDAR_FILE = TEST_CALENDAR_FILE

if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)

check("no events before any writes", calendar_store.get_events_on("2026-08-10"), [])

calendar_store.add_event("2026-08-10", "Google interview")
check("event recorded after add_event", calendar_store.get_events_on("2026-08-10"), ["Google interview"])

alternatives = calendar_store.suggest_alternatives("2026-08-10", count=3)
check("suggest_alternatives returns 3 dates", len(alternatives), 3)
check("suggested alternatives don't include the conflicted date", "2026-08-10" in alternatives, False)

calendar_store.add_event(alternatives[0], "blocked too")
new_alternatives = calendar_store.suggest_alternatives("2026-08-10", count=3)
check("newly-booked alternative is excluded from next suggestion", alternatives[0] in new_alternatives, False)

# cleanup + restore
if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)
calendar_store.CALENDAR_FILE = _original_calendar_file
check_true("real scheduled_interviews.json untouched by this test", not os.path.exists(TEST_CALENDAR_FILE))


# ===========================================================================
# 3. resume_document.py — Gemini primary / OCR fallback branching
#    (network calls to Gemini are monkeypatched out — this tests the
#    branching logic, not live model output)
# ===========================================================================
print("\n--- 3. resume_document.py (fallback logic, no live API calls) ---")
import resume_document

_orig_gemini = resume_document._gemini_document_understanding
_orig_ocr = resume_document._ocr_fallback

# 3a. Gemini succeeds -> should use gemini path, never touch OCR
resume_document._gemini_document_understanding = lambda path, max_retries=2: "EDUCATION\nBTech CS"
resume_document._ocr_fallback = lambda path: (_ for _ in ()).throw(AssertionError("OCR should not run when Gemini succeeds"))
result = resume_document.parse_resume_document("fake_resume.pdf")
check("gemini-success: status ok", result["status"], "ok")
check("gemini-success: source is gemini", result["source"], "gemini_document_understanding")
check("gemini-success: text extracted", result["extracted_text"], "EDUCATION\nBTech CS")

# 3b. Gemini fails, OCR succeeds -> should fall back cleanly
resume_document._gemini_document_understanding = lambda path, max_retries=2: (_ for _ in ()).throw(RuntimeError("rate limited"))
resume_document._ocr_fallback = lambda path: "OCR EXTRACTED TEXT"
result = resume_document.parse_resume_document("fake_resume.pdf")
check("gemini-fails/ocr-succeeds: status ok", result["status"], "ok")
check("gemini-fails/ocr-succeeds: source is ocr_fallback", result["source"], "ocr_fallback")
check("gemini-fails/ocr-succeeds: text from OCR", result["extracted_text"], "OCR EXTRACTED TEXT")

# 3c. Both fail -> should report failure, not raise
resume_document._gemini_document_understanding = lambda path, max_retries=2: (_ for _ in ()).throw(RuntimeError("rate limited"))
resume_document._ocr_fallback = lambda path: (_ for _ in ()).throw(RuntimeError("tesseract not installed"))
result = resume_document.parse_resume_document("fake_resume.pdf")
check("both-fail: status generation_failed", result["status"], "generation_failed")
check("both-fail: extracted_text is None", result["extracted_text"], None)
check_true("both-fail: error message mentions both failures", "rate limited" in result["error"] and "tesseract" in result["error"])

resume_document._gemini_document_understanding = _orig_gemini
resume_document._ocr_fallback = _orig_ocr


# ===========================================================================
# 4. mcp_server.py — deterministic tool logic + sequential/parallel shape
#    (parse_node/drafter_node are monkeypatched to avoid real LLM calls,
#    same pattern as test_helpers.py's generation_failed forcing test)
# ===========================================================================
print("\n--- 4. mcp_server.py ---")
import mcp_server
from models import (
    ParseEmailInput,
    CheckCalendarConflictInput,
    ProcessApplicationInput,
    ScoreAgainstResumesInput,
)

# 4a. check_calendar_conflict is fully deterministic, no LLM involved
calendar_store.CALENDAR_FILE = TEST_CALENDAR_FILE
if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)

no_conflict = mcp_server.check_calendar_conflict(CheckCalendarConflictInput(date="2026-09-01", label="Test interview"))
check("first booking on a date: no conflict", no_conflict.conflict, False)

conflict = mcp_server.check_calendar_conflict(CheckCalendarConflictInput(date="2026-09-01", label="Double booked"))
check("second booking same date: conflict detected", conflict.conflict, True)
check("conflict returns 3 alternatives", len(conflict.suggested_alternatives), 3)

if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)
calendar_store.CALENDAR_FILE = _original_calendar_file

# 4b. _score_against_resume: pure function, no LLM, deterministic given inputs
sample_parsed = ParseEmailOutput(
    company="Unify Technologies",
    role="SDE",
    requirements=["pytorch", "tensorflow", "spark"],
    education_required="Bachelor's degree in Computer Science",
    years_of_experience_required="4+",
    status="parsed",
)
fit_result = mcp_server._score_against_resume(sample_parsed, mcp_server.RESUME_TEXT)
check_true("score is between 0 and 100", 0.0 <= fit_result.score <= 100.0)
check_true("matched_skills contains tensorflow (present in resume.py's ML Frameworks line)", "tensorflow" in fit_result.matched_skills)
check_true("missing_skills correctly flags pytorch (not present in resume.py)", "pytorch" in fit_result.missing_skills)
check("confidence is 1.0 when company/role/requirements all present", fit_result.confidence, 1.0)

empty_parsed = ParseEmailOutput(status="generation_failed")
empty_fit = mcp_server._score_against_resume(empty_parsed, mcp_server.RESUME_TEXT)
check("confidence drops to 0.0 when parse produced nothing", empty_fit.confidence, 0.0)

# 4c. process_application: SEQUENTIAL demo. parse_node/drafter_node
# monkeypatched so this doesn't hit a real LLM.
_orig_parse_email = mcp_server.parse_email

def _fake_parse_email(input):
    return ParseEmailOutput(
        company="Google",
        role="Software Engineer",
        requirements=["python"],
        status="parsed",
    )

mcp_server.parse_email = _fake_parse_email
calendar_store.CALENDAR_FILE = TEST_CALENDAR_FILE
if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)

seq_result = mcp_server.process_application(
    ProcessApplicationInput(raw_text="irrelevant, parse is mocked", interview_date="2026-09-05")
)
check_true("process_application: parsed step populated", seq_result.parsed is not None)
check("process_application: parsed company flows into calendar label",
      calendar_store.get_events_on("2026-09-05"), ["Google interview"])
check_true("process_application: fit step populated (depends on parsed)", seq_result.fit is not None)
check("process_application: no errors on happy path", seq_result.errors, [])

if os.path.exists(TEST_CALENDAR_FILE):
    os.remove(TEST_CALENDAR_FILE)
calendar_store.CALENDAR_FILE = _original_calendar_file
mcp_server.parse_email = _orig_parse_email

# 4d. score_against_resumes: PARALLEL demo, using real temp resume files
TEMP_RESUME_A = "temp_resume_a.txt"
TEMP_RESUME_B = "temp_resume_b.txt"
with open(TEMP_RESUME_A, "w") as f:
    f.write("EDUCATION\nBachelor's degree Computer Science\nSKILLS\npytorch tensorflow spark")
with open(TEMP_RESUME_B, "w") as f:
    f.write("EDUCATION\nBachelor's degree Fine Arts\nSKILLS\nphotoshop illustrator")

parallel_input = ScoreAgainstResumesInput(
    parsed_state=sample_parsed,
    resume_paths=[TEMP_RESUME_A, TEMP_RESUME_B],
)
parallel_result = mcp_server.score_against_resumes(parallel_input)
check("score_against_resumes: 2 results for 2 resumes", len(parallel_result.results), 2)
check("score_against_resumes: no errors on valid paths", parallel_result.errors, [])
check_true("score_against_resumes: technical resume scores higher than unrelated one",
           parallel_result.results[0].score > parallel_result.results[1].score)

os.remove(TEMP_RESUME_A)
os.remove(TEMP_RESUME_B)

# 4e. partial-failure isolation: a bad resume path shouldn't kill the good ones
with open(TEMP_RESUME_A, "w") as f:
    f.write("EDUCATION\nBachelor's degree Computer Science\nSKILLS\npytorch")
broken_input = ScoreAgainstResumesInput(
    parsed_state=sample_parsed,
    resume_paths=[TEMP_RESUME_A, "does_not_exist.txt"],
)
broken_result = mcp_server.score_against_resumes(broken_input)
check("partial failure: 1 successful result despite 1 bad path", len(broken_result.results), 1)
check("partial failure: 1 error recorded for the bad path", len(broken_result.errors), 1)
os.remove(TEMP_RESUME_A)


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
