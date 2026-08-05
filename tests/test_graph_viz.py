"""
tests/test_graph_viz.py

Tests graph_viz.py's two functions:
  - get_execution_path(): parsing logic tested against a fake checkpoint
    history (mimicking LangGraph's StateSnapshot shape), zero real API
    calls or real graph execution needed.
  - save_static_diagram(): its fallback behavior when PNG rendering fails
    (mermaid.ink unreachable/erroring), also mocked.

Plus the standard integrity check: this phase is a pure addition, nothing
existing should have changed.
"""
import __init__
from dotenv import load_dotenv
load_dotenv()

import hashlib
import os
from types import SimpleNamespace

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
print("--- 0. Integrity check (graph_viz.py phase: pure addition) ---")

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
        actual_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    check(f"{filename} unchanged", actual_hash, expected_hash)


# ===========================================================================
# 1. get_execution_path() — fake checkpoint history, no real graph run
# ===========================================================================
print("\n--- 1. get_execution_path() (mocked checkpoint history) ---")
import graph_viz


def _fake_snapshot(writes):
    """Mimics the piece of LangGraph's StateSnapshot that
    get_execution_path() actually reads: a .metadata dict with a
    'writes' key mapping node name -> its output."""
    return SimpleNamespace(metadata={"writes": writes} if writes else {})


# get_state_history() returns newest-first; get_execution_path() reverses
# it. Simulate a 4-node run: parse -> memory_lookup -> score -> plan,
# plus the initial empty pre-run snapshot LangGraph always includes.
fake_history_newest_first = [
    _fake_snapshot({"plan": {"needs_followup": True}}),
    _fake_snapshot({"score": {"fit_score": 80.0}}),
    _fake_snapshot({"memory_lookup": {"past_company_notes": []}}),
    _fake_snapshot({"parse": {"company": "Acme"}}),
    _fake_snapshot(None),  # initial snapshot, no writes yet
]

orig_get_state_history = graph_viz.app.get_state_history
graph_viz.app.get_state_history = lambda config: iter(fake_history_newest_first)

path = graph_viz.get_execution_path("fake-thread-1")
graph_viz.app.get_state_history = orig_get_state_history

check("execution path is in correct oldest-first order", path, ["parse", "memory_lookup", "score", "plan"])

# empty history -> empty path, not a crash
graph_viz.app.get_state_history = lambda config: iter([])
empty_path = graph_viz.get_execution_path("never-run-thread")
graph_viz.app.get_state_history = orig_get_state_history
check("unknown thread_id returns an empty path, not an error", empty_path, [])

# print_execution_path() doesn't raise on either case (smoke test)
graph_viz.app.get_state_history = lambda config: iter(fake_history_newest_first)
try:
    graph_viz.print_execution_path("fake-thread-1")
    print_ok = True
except Exception as e:
    print_ok = False
    print(f"    print_execution_path raised: {e}")
graph_viz.app.get_state_history = orig_get_state_history
check_true("print_execution_path() runs without raising", print_ok)


# ===========================================================================
# 2. save_static_diagram() — mocked graph object, no real PNG rendering
# ===========================================================================
print("\n--- 2. save_static_diagram() fallback behavior (mocked) ---")

TEST_PNG_PATH = "test_graph_diagram.png"
TEST_MMD_PATH = "test_graph_diagram.mmd"
for p in (TEST_PNG_PATH, TEST_MMD_PATH):
    if os.path.exists(p):
        os.remove(p)


class _FakeGraphSuccess:
    def draw_mermaid_png(self):
        return b"fake-png-bytes"


class _FakeGraphFailure:
    def draw_mermaid_png(self):
        raise RuntimeError("simulated mermaid.ink network failure")

    def draw_mermaid(self):
        return "graph TD\n  parse --> score"


orig_get_graph = graph_viz.app.get_graph

# 2a. success path: PNG written as-is
graph_viz.app.get_graph = lambda: _FakeGraphSuccess()
result_path = graph_viz.save_static_diagram(TEST_PNG_PATH)
check("success case returns the PNG path", result_path, TEST_PNG_PATH)
check_true("PNG file was actually written", os.path.exists(TEST_PNG_PATH))
with open(TEST_PNG_PATH, "rb") as f:
    check("PNG file contains the rendered bytes", f.read(), b"fake-png-bytes")

# 2b. failure path: falls back to .mmd, doesn't raise
graph_viz.app.get_graph = lambda: _FakeGraphFailure()
fallback_result_path = graph_viz.save_static_diagram(TEST_PNG_PATH)
check("failure case falls back to .mmd path", fallback_result_path, TEST_MMD_PATH)
check_true(".mmd fallback file was actually written", os.path.exists(TEST_MMD_PATH))
with open(TEST_MMD_PATH) as f:
    check_true("mmd fallback contains the raw Mermaid source", "graph TD" in f.read())

graph_viz.app.get_graph = orig_get_graph

for p in (TEST_PNG_PATH, TEST_MMD_PATH):
    if os.path.exists(p):
        os.remove(p)


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
