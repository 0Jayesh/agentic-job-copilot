"""
test_mlflow_setup.py
 
Tests Phase 9: mlflow_setup.py (mlflow.langchain.autolog(), spec 3.9) and
its one-call wiring into graph_builder.py.
 
Uses an isolated SQLite tracking store (test_mlflow.db) and a throwaway
experiment name, never the real mlflow.db a live run would create.
 
Section 2 proves autolog actually captures something, not just that
enable_tracing() returns True without error -- it runs a real (API-key-free)
LangChain Runnable and confirms a trace landed with OK status. MLflow logs
traces asynchronously, so this explicitly flushes the async queue before
querying -- without that, the check races the write and can read as 0
traces even when tracing is working correctly.
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
# 0. INTEGRITY CHECK — graph_builder.py deliberately changed this phase,
#    everything else (including nodes.py, unchanged since Phase 8) is not.
# ===========================================================================
print("--- 0. Integrity check (Phase 9: graph_builder.py intentionally changed) ---")
 
_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_roots = [os.getcwd(), _this_dir, os.path.dirname(_this_dir)]
REPO_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "nodes.py"))),
    _this_dir,
)
 
EXPECTED_HASHES = {
    "nodes.py": "114af420bd9c30d9f744160d34bc3b9966acd0e7c38d5c6aae4eaaf66bbcb51f",
    "state.py": "11f074992b03d9ffbb908e4de1722a660e74699ec6eb3f45a406c4dc09f6a6f2",
    "memory.py": "182a259474798ccafa5efbad19f9a5b6c7401925dbdcfe63e016d51e957cdfce",
    "education_maps.py": "803024213b617f60e809ec8e045ad0194770d1e91040c7daf291817fd1170d3e",
    "resume.py": "445070b71550abf2efb453521876cb75061855dfdc772860ddf5d637e64868a8",
    # Changed this phase: 3 lines added (import mlflow_setup + call
    # enable_tracing()) before score_node_wrapper is defined. Nothing else
    # in the file touched.
    "graph_builder.py": "788e8be424c4ae3399981d6df78cb1735f8237a049357cb341d512b43ee788ed",
    "calendar_store.py": "b886b81b0ca870ad54c3a19053430dd6252c6560027ea52346af6d4a4a59706b",
    "mcp_server.py": "23685fbfa552f0da5f8382dfa10c7b6dc179a1f338c93882e68d953e055af6a7",
    "models.py": "8f47510dcd3424969d234e84427ae85c56abb3748d655325b17eea55ccac6476",
    "resume_document.py": "d0b80690cd6c138d423805cfbb506449a678cea000c965f69fdcfa2ed0331fe0",
}
 
for filename, expected_hash in EXPECTED_HASHES.items():
    full_path = os.path.join(REPO_ROOT, filename)
    if not os.path.exists(full_path):
        check(f"{filename} exists", False, True)
        continue
    with open(full_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    check(f"{filename} matches expected state", actual_hash, expected_hash)
 
 
# ===========================================================================
# 1. mlflow_setup.py — standalone, isolated tracking store
# ===========================================================================
print("\n--- 1. mlflow_setup.py ---")
import mlflow_setup
 
TEST_DB = "test_mlflow.db"
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)
 
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{TEST_DB}"
mlflow_setup._tracing_enabled = False  # reset the module-level idempotency flag for this test
 
result = mlflow_setup.enable_tracing(experiment_name="test-job-search-copilot")
check("enable_tracing() returns True on first call", result, True)
 
result_again = mlflow_setup.enable_tracing(experiment_name="test-job-search-copilot")
check("enable_tracing() is idempotent (returns True without re-enabling)", result_again, True)
 
 
# ===========================================================================
# 2. Autolog actually captures a trace — not just that setup succeeds
# ===========================================================================
print("\n--- 2. Autolog captures a real trace ---")
from langchain_core.runnables import RunnableLambda
import mlflow
 
chain = RunnableLambda(lambda x: f"processed: {x}")
chain_result = chain.invoke("test input")
check("test chain runs correctly", chain_result, "processed: test input")
 
mlflow.flush_trace_async_logging(terminate=False)
 
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("test-job-search-copilot")
check_true("test experiment was created", exp is not None)
 
traces = client.search_traces(experiment_ids=[exp.experiment_id])
check_true("at least one trace was captured", len(traces) >= 1)
if traces:
    check("captured trace has OK status", str(traces[0].info.status), "TraceStatus.OK")
 
 
# cleanup — mlflow's SQLite backend can hold a connection open on Windows,
# same as chromadb's PersistentClient did earlier. Treat a leftover file as
# a warning, not a failure.
import gc
gc.collect()
try:
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
except PermissionError as e:
    print(f"[WARN] Could not remove {TEST_DB} (Windows file lock on mlflow's sqlite file) -- not a functional failure: {e}")
if os.environ.get("MLFLOW_TRACKING_URI") == f"sqlite:///{TEST_DB}":
    del os.environ["MLFLOW_TRACKING_URI"]
 
 
# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)