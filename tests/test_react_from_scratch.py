"""
tests/test_react_from_scratch.py

Tests react_from_scratch.py's control-flow logic with a scripted sequence
of fake LLM responses (invoke_with_fallback mocked), so this runs with
zero real API calls -- distinct from actually running
`python react_from_scratch.py`, which deliberately DOES call real
Gemini/Groq to prove the loop works against a live model.

Plus the standard integrity check, now also covering graph_viz.py from the
previous phase.
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
# 0. INTEGRITY CHECK — pure addition, nothing existing touched
# ===========================================================================
print("--- 0. Integrity check (react_from_scratch.py phase: pure addition) ---")

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
    "graph_viz.py": "ff4e694684db4a62bc9502da7648a4472b21778cfe7227bdfc91023996768a1b",
}

for filename, expected_hash in EXPECTED_HASHES.items():
    full_path = os.path.join(REPO_ROOT, filename)
    if not os.path.exists(full_path):
        check(f"{filename} exists", False, True)
        continue
    with open(full_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    check(f"{filename} unchanged", actual_hash, expected_hash)


# ===========================================================================
# 1. parse_step() — pure string parsing, no API calls
# ===========================================================================
print("\n--- 1. parse_step() ---")
import react_from_scratch as rfs

action_step = rfs.parse_step(
    "Thought: I should score this job.\nAction: match_resume\nAction Input: Some job text here"
)
check("action correctly extracted", action_step["action"], "match_resume")
check("action_input correctly extracted", action_step["action_input"], "Some job text here")
check("final_answer is None when not present", action_step["final_answer"], None)

final_step = rfs.parse_step("Thought: I'm done.\nFinal Answer: You should apply, great fit.")
check("final_answer correctly extracted", final_step["final_answer"], "You should apply, great fit.")
check("action is None when not present", final_step["action"], None)

garbage_step = rfs.parse_step("I don't know what format you want.")
check("malformed text: action is None", garbage_step["action"], None)
check("malformed text: final_answer is None", garbage_step["final_answer"], None)


# ===========================================================================
# 2. match_resume() tool — mocked parse_node/score_node, tests the
#    formatting logic, not the LLM
# ===========================================================================
print("\n--- 2. match_resume() tool (mocked parse_node/score_node) ---")
_orig_parse_node = rfs.parse_node
_orig_score_node = rfs.score_node

rfs.parse_node = lambda state: {"company": "TestCo", "role": "Engineer"}
rfs.score_node = lambda state, resume_text: {
    "fit_score": 77.5,
    "matched_skills": ["python", "tensorflow"],
    "missing_skills": ["spark"],
}
observation = rfs.match_resume("irrelevant, pipeline is mocked")
rfs.parse_node = _orig_parse_node
rfs.score_node = _orig_score_node

check_true("observation mentions the fit score", "77.5" in observation)
check_true("observation mentions matched skills", "python" in observation and "tensorflow" in observation)
check_true("observation mentions missing skills", "spark" in observation)


# ===========================================================================
# 3. run_react_loop() control flow — scripted fake LLM responses
# ===========================================================================
print("\n--- 3. run_react_loop() (scripted fake LLM responses, no real calls) ---")


def _mock_llm_sequence(responses):
    """Returns a stand-in for invoke_with_fallback that yields each
    response in order on successive calls."""
    it = iter(responses)

    def _fake(prompt):
        return next(it)

    return _fake


_orig_invoke = rfs.invoke_with_fallback
_orig_match_resume = rfs.match_resume

# 3a. happy path: one action, one observation, then a final answer
rfs.invoke_with_fallback = _mock_llm_sequence(
    [
        "Thought: let me score it.\nAction: match_resume\nAction Input: some JD text",
        "Thought: got the score.\nFinal Answer: Great fit, you should apply.",
    ]
)
rfs.match_resume = lambda job_description: "Fit score: 90. Matched: [python]. Missing: []."
result = rfs.run_react_loop("Should I apply to this job?", verbose=False)
check("happy path returns the final answer", result, "Great fit, you should apply.")

# 3b. malformed first response -> stops cleanly instead of looping forever
rfs.invoke_with_fallback = _mock_llm_sequence(["I refuse to follow the format."])
result_malformed = rfs.run_react_loop("test query", verbose=False)
check_true("malformed response stops with a clear message", result_malformed.startswith("Stopped: could not parse"))

# 3c. never produces a Final Answer -> stops at MAX_ITERATIONS, doesn't hang
infinite_action = "Thought: still working.\nAction: match_resume\nAction Input: x"
rfs.invoke_with_fallback = _mock_llm_sequence([infinite_action] * rfs.MAX_ITERATIONS)
result_maxed = rfs.run_react_loop("test query", verbose=False)
check("exceeding max iterations stops with a clear message", result_maxed, "Stopped: exceeded max iterations without a Final Answer.")

rfs.invoke_with_fallback = _orig_invoke
rfs.match_resume = _orig_match_resume


# ===========================================================================
# 4. run_react_loop_steps() -- structured version for the UI, same control
#    flow as run_react_loop() but returns step records instead of printing
# ===========================================================================
print("\n--- 4. run_react_loop_steps() (scripted fake LLM responses) ---")

# 4a. happy path: one action step (with observation), one final-answer step
rfs.invoke_with_fallback = _mock_llm_sequence(
    [
        "Thought: let me score it.\nAction: match_resume\nAction Input: some JD text",
        "Thought: got the score.\nFinal Answer: Great fit, you should apply.",
    ]
)
rfs.match_resume = lambda job_description: "Fit score: 90. Matched: [python]. Missing: []."
result = rfs.run_react_loop_steps("Should I apply to this job?")
rfs.invoke_with_fallback = _orig_invoke
rfs.match_resume = _orig_match_resume

check("happy path returns the final answer", result["final_answer"], "Great fit, you should apply.")
check("happy path has no stopped_reason", result["stopped_reason"], None)
check("happy path recorded exactly 2 steps", len(result["steps"]), 2)
check("step 1 has the action recorded", result["steps"][0]["action"], "match_resume")
check("step 1 has the observation recorded", result["steps"][0]["observation"], "Fit score: 90. Matched: [python]. Missing: [].")
check_true("step 2 (final answer) has no action/observation", result["steps"][1]["action"] is None and result["steps"][1]["observation"] is None)

# 4b. malformed response -> stops cleanly with a stopped_reason, no exception
rfs.invoke_with_fallback = _mock_llm_sequence(["I refuse to follow the format."])
result_malformed = rfs.run_react_loop_steps("test query")
rfs.invoke_with_fallback = _orig_invoke
check("malformed response: final_answer is None", result_malformed["final_answer"], None)
check_true("malformed response: stopped_reason explains why", "could not parse" in result_malformed["stopped_reason"].lower())

# 4c. exceeds max_iterations -> stops cleanly, doesn't hang
rfs.invoke_with_fallback = _mock_llm_sequence([infinite_action] * 3)
result_maxed = rfs.run_react_loop_steps("test query", max_iterations=3)
rfs.invoke_with_fallback = _orig_invoke
check("max-iterations case: final_answer is None", result_maxed["final_answer"], None)
check_true("max-iterations case: stopped_reason explains why", "exceeded max iterations" in result_maxed["stopped_reason"].lower())
check("max-iterations case recorded exactly 3 steps (the max)", len(result_maxed["steps"]), 3)

rfs.invoke_with_fallback = _orig_invoke
rfs.match_resume = _orig_match_resume


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
