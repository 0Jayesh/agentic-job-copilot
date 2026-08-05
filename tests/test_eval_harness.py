"""
tests/test_eval_harness.py

Tests evaluate_case()'s grading logic in isolation, using fake
parse_node/score_node/planner_node so this runs with zero real API calls --
distinct from actually running eval_harness.py against the golden set
(which deliberately DOES call real Gemini/Groq, see eval_harness.py's
docstring). This file checks "does the grader grade correctly", not "is
the LLM's parsing good" -- those are different questions with different
testing strategies, which is the exact distinction spec 3.5 asks for.

Also carries the now-standard integrity check: eval_harness.py and
golden_set.py are pure additions and shouldn't have required touching any
existing file.
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
# 0. INTEGRITY CHECK — this phase adds golden_set.py + eval_harness.py only,
#    no existing file should have changed.
# ===========================================================================
print("--- 0. Integrity check (Phase 10: pure addition, nothing else touched) ---")

_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_roots = [os.getcwd(), _this_dir, os.path.dirname(_this_dir)]
REPO_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "nodes.py"))),
    _this_dir,
)

EXPECTED_HASHES = {
    "nodes.py": "3531645aec66ac9d3df4331bf52735388ce62e2c8eb75f5b2d56eba500d6cfb3",
    "state.py": "df71b28e24d815183891347cf4b1ff238a3e03ad9192105000b41e41bdec8d2d",
    "memory.py": "f51e64d21d01ac869cb9f81df28ddf45134435f31fc1af0284b4066a102f80e2",
    "education_maps.py": "a2e190e5f41bd0e6a80dbe59553832254ff0765dc870d5e7cf0ed2d845651f04",
    "resume.py": "915bb513a4a5f269a39d3acfa75c7a04ccf855bff0595c843fd15b9d14691ff5",
    "graph_builder.py": "835f510a2f1fadc7a70359606c93aa95fdf5aa8faa9a9b5ad75d40e4d973a139",
    "calendar_store.py": "b886b81b0ca870ad54c3a19053430dd6252c6560027ea52346af6d4a4a59706b",
    "mcp_server.py": "23685fbfa552f0da5f8382dfa10c7b6dc179a1f338c93882e68d953e055af6a7",
    "models.py": "8f47510dcd3424969d234e84427ae85c56abb3748d655325b17eea55ccac6476",
    "resume_document.py": "d0b80690cd6c138d423805cfbb506449a678cea000c965f69fdcfa2ed0331fe0",
    # structured_memory.py hash updated: get_company_memory_detailed() now
    # does a case-insensitive substring match instead of exact equality (UI
    # search fix -- memory_lookup_node's exact-match core lookup is untouched).
    "structured_memory.py": "09e69326f59daf936eb40c45f1b559aeba6c5cb7b68b2a770f46ff01961771e5",
    "vector_memory.py": "ab04fad35af7935ff5c9ee0b0f92c0e8a4022331e5d161a8d932daad7d88541b",
}

for filename, expected_hash in EXPECTED_HASHES.items():
    full_path = os.path.join(REPO_ROOT, filename)
    if not os.path.exists(full_path):
        check(f"{filename} exists", False, True)
        continue
    with open(full_path, "rb") as f:
        # Normalize CRLF -> LF before hashing: this repo is edited on Windows
        # (CRLF) but CI runs on a Linux runner, whose git checkout can hand
        # back LF line endings for the exact same content. Hashing raw bytes
        # would make every file falsely appear changed cross-platform even
        # when nothing in the code differs -- normalizing first makes the
        # hash content-only, not whitespace-convention-only.
        actual_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    check(f"{filename} unchanged", actual_hash, expected_hash)


# ===========================================================================
# 1. golden_set.py — structural sanity, no API calls
# ===========================================================================
print("\n--- 1. golden_set.py structure ---")
from golden_set import GOLDEN_SET

check_true("golden set has at least 10 cases", len(GOLDEN_SET) >= 10)
ids = [c["id"] for c in GOLDEN_SET]
check("all case ids are unique", len(ids), len(set(ids)))
check_true("every case has a non-empty raw_input", all(c.get("raw_input") for c in GOLDEN_SET))
check_true("every case has an expected dict", all(isinstance(c.get("expected"), dict) for c in GOLDEN_SET))
check_true(
    "the known-bug case (no_company_mentioned) is present, not avoided",
    any(c["id"] == "no_company_mentioned" for c in GOLDEN_SET),
)


# ===========================================================================
# 2. evaluate_case() grading logic — mocked pipeline, zero real API calls
# ===========================================================================
print("\n--- 2. evaluate_case() grading logic (mocked, no real API calls) ---")
import eval_harness


def _mock(company=None, role=None, fit_score=None, needs_followup=None):
    """Monkeypatches parse_node/score_node/planner_node for exactly one
    evaluate_case() call, then restores them."""
    orig_parse = eval_harness.parse_node
    orig_score = eval_harness.score_node
    orig_plan = eval_harness.planner_node

    eval_harness.parse_node = lambda state: {"company": company, "role": role}
    eval_harness.score_node = lambda state, resume_text: {**state, "fit_score": fit_score}
    eval_harness.planner_node = lambda state: {**state, "needs_followup": needs_followup}

    return orig_parse, orig_score, orig_plan


def _restore(originals):
    eval_harness.parse_node, eval_harness.score_node, eval_harness.planner_node = originals


# 2a. everything matches -> case passes
case = {
    "id": "mock_happy_path",
    "raw_input": "irrelevant, pipeline is mocked",
    "expected": {
        "company_contains": "Google",
        "role_contains": "Engineer",
        "fit_score_min": 60.0,
        "fit_score_max": 100.0,
        "expected_needs_followup": True,
    },
}
originals = _mock(company="Google Inc.", role="Software Engineer", fit_score=85.0, needs_followup=True)
result = eval_harness.evaluate_case(case)
_restore(originals)
check("happy-path case passes overall", result["passed"], True)
check_true("all individual checks passed", all(c["passed"] for c in result["checks"]))

# 2b. fit_score out of range -> case fails
case2 = {**case, "id": "mock_bad_score"}
originals = _mock(company="Google Inc.", role="Software Engineer", fit_score=20.0, needs_followup=True)
result2 = eval_harness.evaluate_case(case2)
_restore(originals)
check("out-of-range fit_score fails the case", result2["passed"], False)

# 2c. informational check out of range -> case still passes overall
case3 = {
    "id": "mock_informational",
    "raw_input": "irrelevant",
    "expected": {
        "fit_score_min": 90.0,
        "fit_score_max": 100.0,
        "fit_score_informational": True,
    },
}
originals = _mock(company="X", role="Y", fit_score=10.0, needs_followup=False)
result3 = eval_harness.evaluate_case(case3)
_restore(originals)
check("informational-only failure does not fail the case", result3["passed"], True)
check_true(
    "the informational check itself is recorded as failed (not silently dropped)",
    any(not c["passed"] and c["informational"] for c in result3["checks"]),
)

# 2d. unresolved company correctly detected -> case passes
case4 = {
    "id": "mock_unresolved_correct",
    "raw_input": "irrelevant",
    "expected": {"expect_unresolved_company": True},
}
originals = _mock(company="None", role=None, fit_score=23.0, needs_followup=False)
result4 = eval_harness.evaluate_case(case4)
_restore(originals)
check("correctly-unresolved company passes", result4["passed"], True)

# 2e. company hallucinated when it should have been unresolved -> case fails
case5 = {**case4, "id": "mock_unresolved_violated"}
originals = _mock(company="Definitely A Real Company Inc.", role=None, fit_score=23.0, needs_followup=False)
result5 = eval_harness.evaluate_case(case5)
_restore(originals)
check("hallucinated company when none was stated fails the case", result5["passed"], False)

# 2f. exception inside the pipeline is caught by run_eval(), not left to crash
def _raise(*args, **kwargs):
    raise RuntimeError("simulated pipeline failure")

orig_parse = eval_harness.parse_node
eval_harness.parse_node = _raise
passed, total, results = eval_harness.run_eval(golden_set=[{"id": "mock_crash", "raw_input": "x", "expected": {}}], verbose=False)
eval_harness.parse_node = orig_parse
check("run_eval() catches a raised exception instead of crashing", passed, 0)
check("run_eval() still reports the total case count", total, 1)
check_true("the crashed case's error message is captured", "error" in results[0])


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
