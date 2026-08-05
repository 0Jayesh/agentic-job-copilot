"""
test_memory_layer.py

Tests the memory-phase files:
  - structured_memory.py  (SQLite structured store)
  - vector_memory.py      (Chroma semantic store)
  - migrate_memory.py     (one-time JSON -> SQLite+Chroma migration)

Same style as test_mcp_layer.py: plain check()-based assertions, run
directly with `python test_memory_layer.py`, isolated from real data files
(uses throwaway DB/collection paths, never company_memory.sqlite3 or
chroma_memory/ that a real run would use).

Section 0 extends the untouched-files integrity check from
test_mcp_layer.py to also cover the Phase 6 files (calendar_store.py,
mcp_server.py, models.py, resume_document.py) -- this phase shouldn't touch
those either, on top of the original 6 files from Phase 1-5.

Neither memory.py nor nodes.py nor graph_builder.py are modified by this
phase -- structured_memory.py and vector_memory.py are new, parallel
stores, not replacements wired into the graph yet.
"""
import __init__
from dotenv import load_dotenv
load_dotenv()

import hashlib
import json
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
# 0. INTEGRITY CHECK — original 6 files + the 4 Phase 6 files, all untouched
# ===========================================================================
print("--- 0. Untouched-files integrity check (Phase 1-6 files) ---")

_this_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_roots = [os.getcwd(), _this_dir, os.path.dirname(_this_dir)]
REPO_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "nodes.py"))),
    _this_dir,
)

EXPECTED_HASHES = {
    # Phase 1-5 -- never touched, checked since the MCP phase
    # nodes.py hash updated in Phase 8: memory_lookup_node/finalize_node now
    # import from structured_memory.py/vector_memory.py instead of memory.py.
    "nodes.py": "3531645aec66ac9d3df4331bf52735388ce62e2c8eb75f5b2d56eba500d6cfb3",
    "state.py": "df71b28e24d815183891347cf4b1ff238a3e03ad9192105000b41e41bdec8d2d",
    "memory.py": "f51e64d21d01ac869cb9f81df28ddf45134435f31fc1af0284b4066a102f80e2",
    "education_maps.py": "a2e190e5f41bd0e6a80dbe59553832254ff0765dc870d5e7cf0ed2d845651f04",
    "resume.py": "915bb513a4a5f269a39d3acfa75c7a04ccf855bff0595c843fd15b9d14691ff5",
    # graph_builder.py hash updated in Phase 9: calls mlflow_setup.enable_tracing()
    # once at graph build time (mlflow.langchain.autolog()) -- 3 lines added, nothing else changed.
    "graph_builder.py": "835f510a2f1fadc7a70359606c93aa95fdf5aa8faa9a9b5ad75d40e4d973a139",
    # Phase 6 -- committed, should now also stay untouched
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
        # Normalize CRLF -> LF before hashing: this repo is edited on Windows
        # (CRLF) but CI runs on a Linux runner, whose git checkout can hand
        # back LF line endings for the exact same content. Hashing raw bytes
        # would make every file falsely appear changed cross-platform even
        # when nothing in the code differs -- normalizing first makes the
        # hash content-only, not whitespace-convention-only.
        actual_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    check(f"{filename} unchanged", actual_hash, expected_hash)


# ===========================================================================
# 1. structured_memory.py — isolated test DB, never company_memory.sqlite3
# ===========================================================================
print("\n--- 1. structured_memory.py (isolated test DB) ---")
import structured_memory

TEST_DB_FILE = "test_company_memory.sqlite3"
_original_db_file = structured_memory.DB_FILE
structured_memory.DB_FILE = TEST_DB_FILE
if os.path.exists(TEST_DB_FILE):
    os.remove(TEST_DB_FILE)

structured_memory.init_db()

# 1a. basic save/get, drop-in compatible shape with memory.py's functions
check("no notes before any writes", structured_memory.get_company_memory("TestCorp"), [])

structured_memory.save_company_memory("TestCorp", "Interviewed, rejected - lacked Spark", fit_score=45.0, status="rejected")
structured_memory.save_company_memory("TestCorp", "Reached out again, no response", fit_score=None, status=None)
notes = structured_memory.get_company_memory("TestCorp")
check("2 notes recorded for TestCorp", len(notes), 2)
check("notes are plain strings (memory.py-compatible shape)", all(isinstance(n, str) for n in notes), True)

check("different company has no notes", structured_memory.get_company_memory("OtherCorp"), [])

# 1b. the guard: None/Not mentioned/empty never create a real company row
row_id = structured_memory.save_company_memory("None", "Fit score 23.0, status: no_action_needed", fit_score=23.0, status="no_action_needed")
check_true("unresolved company write returns a row id", row_id is not None)
check("unresolved company never appears in company_notes", structured_memory.get_company_memory("None"), [])

structured_memory.save_company_memory("Not mentioned", "Fit score 23.0, status: no_action_needed")
structured_memory.save_company_memory(None, "totally missing company")
unresolved = structured_memory.get_unresolved_notes()
check("3 notes correctly routed to unresolved_company_notes", len(unresolved), 3)
check_true(
    "unresolved notes preserve the raw (garbage) company value for inspection",
    any(u["raw_company_value"] == "None" for u in unresolved)
    and any(u["raw_company_value"] == "Not mentioned" for u in unresolved),
)

# 1c. structured querying — the actual point of moving off JSON
structured_memory.save_company_memory("LowFitCo", "auto-scored", fit_score=30.0, status="no_action_needed")
structured_memory.save_company_memory("HighFitCo", "auto-scored", fit_score=90.0, status="pending_approval")
low_fit = structured_memory.query_low_fit_no_followup(threshold=60.0)
low_fit_companies = {row["company"] for row in low_fit}
check_true("query_low_fit_no_followup includes LowFitCo", "LowFitCo" in low_fit_companies)
check_true("query_low_fit_no_followup excludes HighFitCo", "HighFitCo" not in low_fit_companies)

# 1d. detailed getter returns structured fields, not just note text
detailed = structured_memory.get_company_memory_detailed("TestCorp")
check("detailed getter returns 2 rows for TestCorp", len(detailed), 2)
check_true("detailed rows carry fit_score/status/created_at", all(
    "fit_score" in row and "status" in row and "created_at" in row for row in detailed
))

# 1e. fuzzy match fix -- reproduces the exact real-world bug observed live:
# the LLM parsed the same real company as "Simelabs" once and
# "Simelabs / Astek" another time, and the old exact-match search silently
# missed one of them depending on what you typed.
structured_memory.save_company_memory("Simelabs", "First run, parsed as just Simelabs", fit_score=86.7, status="approved_ready_to_send")
structured_memory.save_company_memory("Simelabs / Astek", "Second run, parsed with suffix", fit_score=80.0, status="approved_ready_to_send")

exact_search = structured_memory.get_company_memory_detailed("Simelabs / Astek")
check("exact match still works", len(exact_search), 1)

partial_search = structured_memory.get_company_memory_detailed("Simelabs")
check_true(
    "partial search now finds BOTH company-name variants (the actual bug fix)",
    len(partial_search) == 2,
)
found_companies = {row["company"] for row in partial_search}
check("both variants are distinguishable via the returned company field", found_companies, {"Simelabs", "Simelabs / Astek"})

case_insensitive_search = structured_memory.get_company_memory_detailed("simelabs")
check_true("search is case-insensitive too", len(case_insensitive_search) == 2)

unrelated_search = structured_memory.get_company_memory_detailed("TestCorp")
check("unrelated company search is unaffected by the fuzzy-match change", len(unrelated_search), 2)

# cleanup + restore
if os.path.exists(TEST_DB_FILE):
    os.remove(TEST_DB_FILE)
structured_memory.DB_FILE = _original_db_file
check_true("real company_memory.sqlite3 untouched by this test", not os.path.exists(TEST_DB_FILE))


# ===========================================================================
# 2. vector_memory.py — requires chromadb + nodes.py's embedder to import.
#    Kept as a distinct, clearly-separated section so a failure here
#    doesn't hide failures in section 1 or 3.
# ===========================================================================
print("\n--- 2. vector_memory.py (isolated Chroma collection) ---")
try:
    import vector_memory

    TEST_PERSIST_DIR = "test_chroma_memory"
    _original_persist_dir = vector_memory.PERSIST_DIR
    _original_collection_name = vector_memory.COLLECTION_NAME
    vector_memory.PERSIST_DIR = TEST_PERSIST_DIR
    vector_memory.COLLECTION_NAME = "test_company_memory"
    vector_memory._client = None
    vector_memory._collection = None

    vector_memory.add_memory("Acme Corp", "Rejected for missing Kubernetes experience", fit_score=40.0, status="rejected", sql_row_id=1)
    vector_memory.add_memory("Acme Corp", "Second application, same gap flagged again", fit_score=42.0, status="rejected", sql_row_id=2)
    vector_memory.add_memory("BetaSoft", "Great culture fit, offer extended", fit_score=95.0, status="approved_ready_to_send", sql_row_id=3)

    by_company = vector_memory.query_by_company("Acme Corp")
    check("query_by_company returns both Acme Corp notes", len(by_company), 2)

    similar = vector_memory.query_similar("missing Kubernetes skills", n_results=3)
    check_true("query_similar returns at least one result", len(similar) >= 1)
    check_true(
        "query_similar's top result is semantically about the Kubernetes gap, not the offer",
        similar[0]["note"] != "Great culture fit, offer extended",
    )

    # cleanup — chromadb's PersistentClient can hold a file handle open on
    # Windows, so rmtree can fail there even though the functional checks
    # above already passed. Drop references, force a GC pass, and treat a
    # leftover directory as a warning, not a failure — it's a Windows file-
    # lock cleanup quirk, not a sign the store itself is broken.
    import gc
    import shutil
    vector_memory._client = None
    vector_memory._collection = None
    gc.collect()
    try:
        if os.path.exists(TEST_PERSIST_DIR):
            shutil.rmtree(TEST_PERSIST_DIR)
    except PermissionError as e:
        print(f"[WARN] Could not remove {TEST_PERSIST_DIR} (Windows file lock on chromadb's sqlite file) -- not a functional failure. Delete it manually if it lingers: {e}")
    vector_memory.PERSIST_DIR = _original_persist_dir
    vector_memory.COLLECTION_NAME = _original_collection_name

except ImportError as e:
    print(f"[SKIPPED] vector_memory.py section — import failed in this environment: {e}")
    print("[SKIPPED] Needs chromadb installed and nodes.py's full dependency chain available.")


# ===========================================================================
# 3. migrate_memory.py — parsing + routing logic, isolated test fixtures
# ===========================================================================
print("\n--- 3. migrate_memory.py ---")
try:
    import migrate_memory

    check(
        "_parse_note extracts score and status",
        migrate_memory._parse_note("Fit score 88.0, status: pending_approval"),
        (88.0, "pending_approval"),
    )
    check(
        "_parse_note returns (None, None) for free-text notes",
        migrate_memory._parse_note("Interviewed March 2026, rejected - lacked Spark experience"),
        (None, None),
    )

    # isolated fixture: throwaway JSON, throwaway SQLite, throwaway Chroma dir
    TEST_JSON = "test_migration_source.json"
    TEST_MIG_DB = "test_migration.sqlite3"
    TEST_MIG_CHROMA = "test_migration_chroma"

    fixture = {
        "RealCo": ["Fit score 75.0, status: pending_approval"],
        "None": ["Fit score 23.0, status: no_action_needed"],
        "Not mentioned": ["Fit score 23.0, status: no_action_needed"],
    }
    with open(TEST_JSON, "w") as f:
        json.dump(fixture, f)

    _orig_old_file = migrate_memory.OLD_MEMORY_FILE
    _orig_db = structured_memory.DB_FILE
    _orig_persist = vector_memory.PERSIST_DIR if "vector_memory" in dir() else None

    migrate_memory.OLD_MEMORY_FILE = TEST_JSON
    structured_memory.DB_FILE = TEST_MIG_DB
    if "vector_memory" in dir():
        vector_memory.PERSIST_DIR = TEST_MIG_CHROMA
        vector_memory._client = None
        vector_memory._collection = None
    if os.path.exists(TEST_MIG_DB):
        os.remove(TEST_MIG_DB)

    migrate_memory.migrate(dedupe=False, dry_run=False)

    check("real company migrated into company_notes", structured_memory.get_company_memory("RealCo"), ["Fit score 75.0, status: pending_approval"])
    unresolved_after_migration = structured_memory.get_unresolved_notes()
    check("both garbage keys routed to unresolved, not company_notes", len(unresolved_after_migration), 2)
    check("garbage keys did NOT get a company_notes entry", structured_memory.get_company_memory("None"), [])

    # cleanup
    for f in (TEST_JSON, TEST_MIG_DB):
        if os.path.exists(f):
            os.remove(f)
    if "vector_memory" in dir():
        import gc
        import shutil
        vector_memory._client = None
        vector_memory._collection = None
        gc.collect()
        try:
            if os.path.exists(TEST_MIG_CHROMA):
                shutil.rmtree(TEST_MIG_CHROMA)
        except PermissionError as e:
            print(f"[WARN] Could not remove {TEST_MIG_CHROMA} (Windows file lock on chromadb's sqlite file) -- not a functional failure: {e}")
        vector_memory.PERSIST_DIR = _orig_persist
    migrate_memory.OLD_MEMORY_FILE = _orig_old_file
    structured_memory.DB_FILE = _orig_db

except ImportError as e:
    print(f"[SKIPPED] migrate_memory.py section — import failed in this environment: {e}")
    print("[SKIPPED] migrate_memory.py imports vector_memory.py, same dependency chain as section 2.")


# ===========================================================================
print(f"\n=== {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    raise SystemExit(1)
