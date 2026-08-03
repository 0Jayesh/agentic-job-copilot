"""
test_memory_wiring.py

Tests Phase 8: nodes.py's memory_lookup_node and finalize_node now read
and write through structured_memory.py (SQLite) + vector_memory.py
(Chroma) instead of the old memory.py flat-JSON store.

This is the first phase that intentionally changes nodes.py. The change
itself is a 2-line import swap plus one added vector_memory.add_memory()
call inside finalize_node -- see the "nodes.py" entry in EXPECTED_HASHES
below, and the comment on it, for exactly what changed and why.

memory.py itself is left in the repo, untouched, unused -- not deleted, so
the change is easy to revert if needed.
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
# 0. INTEGRITY CHECK — nodes.py deliberately changed (see comment below),
#    everything else (including memory.py, now unused but untouched) is not.
# ===========================================================================
print("--- 0. Integrity check (Phase 8: nodes.py intentionally changed) ---")

_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_roots = [os.getcwd(), _this_dir, os.path.dirname(_this_dir)]
REPO_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "nodes.py"))),
    _this_dir,
)

EXPECTED_HASHES = {
    # Changed in this phase: memory_lookup_node/finalize_node now import
    # from structured_memory.py/vector_memory.py instead of memory.py, and
    # finalize_node also writes to vector_memory. Everything else in the
    # file (parse_node, score_node, planner_node, drafter_node, etc.) is
    # byte-identical to before.
    "nodes.py": "114af420bd9c30d9f744160d34bc3b9966acd0e7c38d5c6aae4eaaf66bbcb51f",

    # Untouched, same as every prior phase
    "state.py": "11f074992b03d9ffbb908e4de1722a660e74699ec6eb3f45a406c4dc09f6a6f2",
    "memory.py": "182a259474798ccafa5efbad19f9a5b6c7401925dbdcfe63e016d51e957cdfce",
    "education_maps.py": "803024213b617f60e809ec8e045ad0194770d1e91040c7daf291817fd1170d3e",
    "resume.py": "445070b71550abf2efb453521876cb75061855dfdc772860ddf5d637e64868a8",
    "graph_builder.py": "b3cc7dceaf6d561c3cb05c68de85305b19d0add2f67a614fe0058f0719b25e82",
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
# 1. finalize_node / memory_lookup_node actually use the new stores
#    (isolated: throwaway SQLite DB + throwaway Chroma collection, never
#    the real company_memory.sqlite3 / chroma_memory/)
# ===========================================================================
print("\n--- 1. Wired memory nodes (isolated stores) ---")
import structured_memory
import vector_memory
from nodes import finalize_node, memory_lookup_node

TEST_DB_FILE = "test_wiring_memory.sqlite3"
TEST_PERSIST_DIR = "test_wiring_chroma"

_orig_db = structured_memory.DB_FILE
_orig_persist = vector_memory.PERSIST_DIR
_orig_collection_name = vector_memory.COLLECTION_NAME

structured_memory.DB_FILE = TEST_DB_FILE
vector_memory.PERSIST_DIR = TEST_PERSIST_DIR
vector_memory.COLLECTION_NAME = "test_wiring_collection"
vector_memory._client = None
vector_memory._collection = None
if os.path.exists(TEST_DB_FILE):
    os.remove(TEST_DB_FILE)

# 1a. finalize_node with a resolved company: should land in both stores
resolved_state = {
    "company": "WiringTestCo",
    "fit_score": 82.0,
    "needs_followup": True,
    "approved": True,
}
result = finalize_node(resolved_state)
check("finalize_node still sets status correctly", result["status"], "approved_ready_to_send")

sql_notes = structured_memory.get_company_memory("WiringTestCo")
check("finalize_node wrote to structured_memory (SQLite)", len(sql_notes), 1)

vector_notes = vector_memory.query_by_company("WiringTestCo")
check("finalize_node also wrote to vector_memory (Chroma)", len(vector_notes), 1)
check_true(
    "the SQLite note and the Chroma note carry the same text",
    sql_notes[0] == vector_notes[0]["note"],
)

# 1b. finalize_node with an unresolved company: SQLite gets it (as
# unresolved), Chroma does NOT -- garbage never gets embedded
unresolved_state = {
    "company": "None",
    "fit_score": 20.0,
    "needs_followup": False,
    "approved": None,
}
finalize_node(unresolved_state)
unresolved_rows = structured_memory.get_unresolved_notes()
check_true("unresolved company landed in unresolved_company_notes", len(unresolved_rows) >= 1)
vector_notes_after = vector_memory.query_by_company("None")
check("unresolved company was NOT written to vector_memory", len(vector_notes_after), 0)

# 1c. memory_lookup_node reads through structured_memory correctly
lookup_result = memory_lookup_node({"company": "WiringTestCo"})
check("memory_lookup_node retrieves the note just written", len(lookup_result["past_company_notes"]), 1)

lookup_new_company = memory_lookup_node({"company": "NeverSeenCo"})
check("memory_lookup_node returns empty list for an unknown company", lookup_new_company["past_company_notes"], [])

# cleanup — same Windows chromadb file-lock handling as the other memory tests
import gc
import shutil
vector_memory._client = None
vector_memory._collection = None
gc.collect()
if os.path.exists(TEST_DB_FILE):
    os.remove(TEST_DB_FILE)
try:
    if os.path.exists(TEST_PERSIST_DIR):
        shutil.rmtree(TEST_PERSIST_DIR)
except PermissionError as e:
    print(f"[WARN] Could not remove {TEST_PERSIST_DIR} (Windows file lock) -- not a functional failure: {e}")

structured_memory.DB_FILE = _orig_db
vector_memory.PERSIST_DIR = _orig_persist
vector_memory.COLLECTION_NAME = _orig_collection_name


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
