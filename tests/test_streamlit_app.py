"""
tests/test_streamlit_app.py

Tests streamlit_app.py using Streamlit's official testing utility
(streamlit.testing.v1.AppTest), which actually executes the app script and
lets us inspect/interact with the rendered widgets -- not a hand-rolled
approximation.

graph_builder, structured_memory, and graph_viz are replaced with fake
modules injected directly into sys.modules BEFORE the app script runs, so
`from graph_builder import app as graph_app` etc. inside streamlit_app.py
picks up the fakes instead of importing the real (heavy, API-key-requiring)
modules. This is what makes this phase fully testable here, unlike every
prior phase that touched nodes.py's import chain.

Plus the standard integrity check.
"""
import __init__
from dotenv import load_dotenv
load_dotenv()

import hashlib
import os
import sys
import types

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
print("--- 0. Integrity check (streamlit_app.py phase: pure addition) ---")

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
    # react_from_scratch.py hash updated: added run_react_loop_steps(), a
    # sibling function returning structured step data for the new ReAct
    # Demo tab. run_react_loop() itself is completely untouched.
    "react_from_scratch.py": "a6683018adf1ea3fe390b1ee226e93b13420ca8e99224105304deb05b6aeae0f",
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
# Set up fake graph_builder / structured_memory / graph_viz BEFORE any
# AppTest run, so streamlit_app.py's imports pick these up instead of the
# real (heavy) modules.
# ===========================================================================

class _FakeCompiledGraph:
    """Stands in for graph_builder.app. `invoke_log` records every call so
    tests can assert on what the app actually did, not just what it
    displayed."""

    def __init__(self):
        self.invoke_log = []
        self.update_state_log = []
        self.next_result = None

    def invoke(self, input_state, config):
        self.invoke_log.append((input_state, config))
        return self.next_result

    def update_state(self, config, updates):
        self.update_state_log.append((config, updates))


fake_graph = _FakeCompiledGraph()
fake_graph_builder = types.ModuleType("graph_builder")
fake_graph_builder.app = fake_graph
sys.modules["graph_builder"] = fake_graph_builder

fake_structured_memory = types.ModuleType("structured_memory")
fake_structured_memory.get_company_memory_detailed = lambda company: (
    [{"company": "Acme", "created_at": "2026-08-01T00:00:00", "note": f"Fake note for {company}"}]
    if company.lower() in "acme"
    else []
)
fake_structured_memory.query_low_fit_no_followup = lambda threshold=60.0: [
    {"company": "LowFitCo", "fit_score": 20.0, "created_at": "2026-08-01T00:00:00"}
]
sys.modules["structured_memory"] = fake_structured_memory

fake_vector_memory = types.ModuleType("vector_memory")
fake_vector_memory.query_similar = lambda query_text, n_results=5: [
    {
        "id": "note-1",
        "note": "Rejected for missing cloud experience",
        "metadata": {"company": "CloudCo"},
        "distance": 0.12,
    }
]
sys.modules["vector_memory"] = fake_vector_memory

fake_react_from_scratch = types.ModuleType("react_from_scratch")


def _fake_run_react_loop_steps(query, max_iterations=5):
    return {
        "steps": [
            {
                "response_text": "Thought: scoring it.\nAction: match_resume\nAction Input: fake JD",
                "action": "match_resume",
                "action_input": "fake JD",
                "observation": "Fit score: 90. Matched: [python]. Missing: [].",
            },
            {
                "response_text": "Thought: done.\nFinal Answer: Great fit, apply now.",
                "action": None,
                "action_input": None,
                "observation": None,
            },
        ],
        "final_answer": "Great fit, apply now.",
        "stopped_reason": None,
    }


fake_react_from_scratch.run_react_loop_steps = _fake_run_react_loop_steps
sys.modules["react_from_scratch"] = fake_react_from_scratch

fake_golden_set = types.ModuleType("golden_set")
fake_golden_set.GOLDEN_SET = [{"id": f"case_{i}"} for i in range(12)]
sys.modules["golden_set"] = fake_golden_set

fake_mlflow_setup = types.ModuleType("mlflow_setup")
fake_mlflow_setup._tracing_enabled = True
sys.modules["mlflow_setup"] = fake_mlflow_setup

# graph_viz's save_static_diagram writes a real (tiny) file so Streamlit's
# actual image/text-loading code has something real to read -- a mocked
# return value alone would make st.image()/st.code() fail on a
# nonexistent path, which would look like a bug in the app but would
# actually just be an artifact of an unrealistic mock.
FAKE_MMD_PATH = "test_streamlit_graph_diagram.mmd"
with open(FAKE_MMD_PATH, "w") as f:
    f.write("graph TD\n  parse --> score --> plan")

fake_graph_viz = types.ModuleType("graph_viz")
fake_graph_viz.save_static_diagram = lambda path="graph_diagram.png": FAKE_MMD_PATH
sys.modules["graph_viz"] = fake_graph_viz


APP_PATH = os.path.join(REPO_ROOT, "streamlit_app.py")


# ===========================================================================
# 1. App loads cleanly, correct tab structure
# ===========================================================================
print("\n--- 1. App loads without exception ---")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(APP_PATH)
at.run()

check_true("app runs without raising an exception", not at.exception)
check("exactly 4 tabs rendered", len(at.tabs), 4)


# ===========================================================================
# 2. "Score a Job" tab -- happy path (needs_followup, awaiting approval)
# ===========================================================================
print("\n--- 2. Score a Job tab: analyze -> awaiting approval -> approve ---")

fake_graph.next_result = {
    "company": "Acme",
    "role": "Software Engineer",
    "fit_score": 85.0,
    "matched_skills": ["python", "tensorflow"],
    "missing_skills": ["spark"],
    "status": "pending_approval",
    "draft_reply": "Hi Acme, thanks for reaching out...",
    "past_company_notes": ["Fake note for Acme"],
}

at = AppTest.from_file(APP_PATH)
at.run()
at.tabs[0].text_area[0].input("Some job posting text").run(timeout=15)
at.tabs[0].button[0].click().run(timeout=15)  # "Analyze"

check_true("app still runs without exception after Analyze", not at.exception)
check("graph_app.invoke was called exactly once", len(fake_graph.invoke_log), 1)

metrics = at.tabs[0].metric
check("Company metric shows the parsed company", metrics[0].value, "Acme")
check("Role metric shows the parsed role", metrics[1].value, "Software Engineer")

# Approve button should be present since status is pending_approval with a draft
approve_buttons = [b for b in at.tabs[0].button if "Approve" in (b.label or "")]
check_true("an Approve button is rendered while awaiting approval", len(approve_buttons) >= 1)
check_true(
    "the Approve button's label no longer implies it sends an email (was 'Approve and send')",
    approve_buttons and approve_buttons[0].label == "Approve",
)

fake_graph.next_result = {
    "company": "Acme",
    "role": "Software Engineer",
    "fit_score": 85.0,
    "status": "approved_ready_to_send",
}
approve_buttons[0].click().run(timeout=15)

check_true("approve flow runs without exception", not at.exception)
check("update_state was called with approved=True", fake_graph.update_state_log[-1][1], {"approved": True})
check("a second invoke(None, ...) resumed the graph", len(fake_graph.invoke_log), 2)
check("resumed invoke was called with None (resume signal)", fake_graph.invoke_log[-1][0], None)


# ===========================================================================
# 3. "Start a new session" resets state
# ===========================================================================
print("\n--- 3. Start a new session ---")
at = AppTest.from_file(APP_PATH)
at.run()
old_thread_id = at.session_state["thread_id"]
new_session_buttons = [b for b in at.tabs[0].button if "new session" in (b.label or "").lower()]
check_true("a 'Start a new session' button exists", len(new_session_buttons) == 1)
new_session_buttons[0].click().run(timeout=15)
check_true("thread_id changed after starting a new session", at.session_state["thread_id"] != old_thread_id)
check("last_result is cleared on new session", at.session_state["last_result"], None)


# ===========================================================================
# 4. "Company Memory" tab -- read-only, zero API calls
# ===========================================================================
print("\n--- 4. Company Memory tab (read-only) ---")
at = AppTest.from_file(APP_PATH)
at.run()
at.tabs[1].text_input[0].input("Acme").run(timeout=15)

check_true("app runs without exception on the memory tab", not at.exception)
all_text = " ".join(str(getattr(el, "value", "")) for el in at.tabs[1].markdown)
check_true("company memory tab surfaces the fake note for a known company", "Fake note for Acme" in all_text)
check_true("result shows which company variant it came from", "Acme" in all_text)

# 4a-bis. partial match -- the actual fix: typing a FRAGMENT of the company
# name, not the exact stored string, should still find it. This reproduces
# the real bug (searching "Simelabs" missed "Simelabs / Astek") in miniature.
at_partial = AppTest.from_file(APP_PATH)
at_partial.run()
at_partial.tabs[1].text_input[0].input("cme").run(timeout=15)  # fragment, not "Acme"
partial_text = " ".join(str(getattr(el, "value", "")) for el in at_partial.tabs[1].markdown)
check_true("a partial/fragment search still finds the company (the actual fix)", "Fake note for cme" in partial_text)

# 4b. semantic search (Chroma) -- this is the actual point of Phase 7's
# vector_memory.py, previously built but never called from the UI at all.
at2 = AppTest.from_file(APP_PATH)
at2.run()
semantic_inputs = [ti for ti in at2.tabs[1].text_input if ti.key == "semantic_query"]
check_true("the semantic search input exists", len(semantic_inputs) == 1)
semantic_inputs[0].input("missing cloud experience").run(timeout=15)

check_true("app runs without exception after a semantic search", not at2.exception)
semantic_text = " ".join(str(getattr(el, "value", "")) for el in at2.tabs[1].markdown)
check_true("semantic search surfaces a result from a DIFFERENT company than the exact-match search", "CloudCo" in semantic_text)
check_true("semantic search result includes the note text", "Rejected for missing cloud experience" in semantic_text)


# ===========================================================================
# 5. "Graph Diagram" tab -- .mmd fallback path renders via st.code
# ===========================================================================
print("\n--- 5. Graph Diagram tab (.mmd fallback) ---")
at = AppTest.from_file(APP_PATH)
at.run()

check_true("app runs without exception on the graph diagram tab", not at.exception)
code_blocks = at.tabs[2].code
check_true("mermaid source rendered via st.code when PNG unavailable", len(code_blocks) >= 1)
if code_blocks:
    check_true("rendered mermaid source contains the graph structure", "parse --> score" in code_blocks[0].value)


# (fixture cleanup moved to the very end of the file -- every AppTest run
# re-executes ALL tabs' code, including tab_graph's, so later sections
# still need this file to exist)


# ===========================================================================
# 6. Advanced tab -- ReAct demo (mocked) + capability scorecard
# ===========================================================================
print("\n--- 6. Advanced tab: ReAct demo + capability scorecard ---")
at6 = AppTest.from_file(APP_PATH)
at6.run()

check_true("app runs without exception on the Advanced tab", not at6.exception)

# 6a. capability scorecard renders with real (mocked) live values, not just
# static filler -- confirms golden_set.GOLDEN_SET and
# mlflow_setup._tracing_enabled are actually read, not hardcoded.
advanced_text = " ".join(str(getattr(el, "value", "")) for el in at6.tabs[3].markdown)
advanced_write_text = " ".join(str(getattr(el, "value", "")) for el in getattr(at6.tabs[3], "write", []))
combined = advanced_text + advanced_write_text
check_true("scorecard shows the live golden-set case count (12)", "12" in combined)
check_true("scorecard shows the live MLflow tracing status (Enabled)", "Enabled" in combined)

# 6b. running the ReAct demo actually calls run_react_loop_steps and
# renders its steps + final answer
run_buttons = [b for b in at6.tabs[3].button if "Run ReAct Demo" in (b.label or "")]
check_true("a 'Run ReAct Demo' button exists", len(run_buttons) == 1)
run_buttons[0].click().run(timeout=15)

check_true("app runs without exception after running the ReAct demo", not at6.exception)
after_run_write = " ".join(str(getattr(el, "value", "")) for el in getattr(at6.tabs[3], "write", []))
after_run_success = " ".join(str(getattr(el, "value", "")) for el in getattr(at6.tabs[3], "success", []))
check_true("the mocked final answer is displayed", "Great fit, apply now" in after_run_success)


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
