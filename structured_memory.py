"""
Structured company memory, backed by SQLite instead of a flat JSON file.

Why SQLite over the JSON approach in memory.py:
  - concurrent-safe writes (JSON had no locking; this project already hit
    duplicate/garbled entries from concurrent test runs once)
  - queryable ("all companies scored below 60") instead of load-whole-file-
    and-filter-in-Python
  - gives Chroma (vector_memory.py) a stable integer ID to key off of, so
    the structured fact and the embedded note stay in sync

This file does NOT modify or replace memory.py — memory.py is left exactly
as-is, still used by the current graph. This is a new, parallel store.
Wiring it into nodes.py/graph_builder.py is a separate, deliberate step
that hasn't happened yet (see migrate_memory.py's docstring).

Storage-boundary guard (new, not present in memory.py): a company value of
None, "", "None", or "not mentioned" (case-insensitive) is never written as
a real company row. It's logged to a separate `unresolved_company_notes`
table instead, so the note isn't silently lost, but it also doesn't
pollute per-company history the way the old JSON store did. This is a
guard at the persistence boundary, not a fix to the upstream parse bug in
nodes.py -- that bug (parse_llm_response_text emitting the literal string
"None"/"Not mentioned") is still open and deliberately untouched.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

DB_FILE = "company_memory.sqlite3"

_UNRESOLVED_MARKERS = {"", "none", "not mentioned", "n/a", "unknown"}


def _is_unresolved_company(company: Optional[str]) -> bool:
    if company is None:
        return True
    return company.strip().lower() in _UNRESOLVED_MARKERS


def is_unresolved_company(company: Optional[str]) -> bool:
    """Public wrapper around the same check save_company_memory/
    get_company_memory use internally -- lets callers (e.g. nodes.py)
    decide whether a company value is real before doing anything company-
    keyed with it, like writing to vector_memory."""
    return _is_unresolved_company(company)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates the tables if they don't exist. Safe to call every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                note TEXT NOT NULL,
                fit_score REAL,
                status TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_company_notes_company
            ON company_notes(company)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unresolved_company_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_company_value TEXT,
                note TEXT NOT NULL,
                fit_score REAL,
                status TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_company_memory(
    company: Optional[str],
    note: str,
    fit_score: Optional[float] = None,
    status: Optional[str] = None,
) -> int:
    """Same call shape as memory.py's save_company_memory (company, note),
    with two optional structured fields added. Returns the new row id.

    Guard: an unresolved company value is redirected to
    unresolved_company_notes instead of creating a fake company history.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        if _is_unresolved_company(company):
            cur = conn.execute(
                """
                INSERT INTO unresolved_company_notes
                    (raw_company_value, note, fit_score, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (company, note, fit_score, status, now),
            )
            return cur.lastrowid

        cur = conn.execute(
            """
            INSERT INTO company_notes (company, note, fit_score, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (company.strip(), note, fit_score, status, now),
        )
        return cur.lastrowid


def get_company_memory(company: Optional[str]) -> List[str]:
    """Same call shape and return shape as memory.py's get_company_memory:
    a list of note strings, most-recent-last, for drop-in compatibility."""
    if _is_unresolved_company(company):
        return []
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT note FROM company_notes
            WHERE company = ?
            ORDER BY id ASC
            """,
            (company.strip(),),
        ).fetchall()
    return [r["note"] for r in rows]


def get_company_memory_detailed(company: str) -> List[dict]:
    """Structured version returning fit_score/status/created_at alongside
    the note, for anything that wants more than the plain note string."""
    if _is_unresolved_company(company):
        return []
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, note, fit_score, status, created_at FROM company_notes
            WHERE company = ?
            ORDER BY id ASC
            """,
            (company.strip(),),
        ).fetchall()
    return [dict(r) for r in rows]


def query_low_fit_no_followup(threshold: float = 60.0) -> List[dict]:
    """Example of the querying SQLite buys over JSON: every note where the
    fit score was below `threshold`. Used for eval/reporting, not by the
    graph itself."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT company, note, fit_score, status, created_at
            FROM company_notes
            WHERE fit_score IS NOT NULL AND fit_score < ?
            ORDER BY created_at DESC
            """,
            (threshold,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_unresolved_notes() -> List[dict]:
    """Everything that would have polluted company_memory.json under the
    old scheme (None/Not mentioned/etc.) -- kept here instead, visible and
    inspectable, not silently discarded."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, raw_company_value, note, fit_score, status, created_at "
            "FROM unresolved_company_notes ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]
