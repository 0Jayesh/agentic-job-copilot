"""
One-time migration: memory.py's flat company_memory.json -> the new
structured_memory.py (SQLite) + vector_memory.py (Chroma) stores.

Run once, manually:
    python migrate_memory.py

What it does NOT do: touch memory.py, nodes.py, or graph_builder.py. After
running this, the graph is still reading/writing through the old
memory.py/company_memory.json exactly as before -- this only backfills the
new stores with what's already there. Wiring the graph itself over to the
new stores is a separate, deliberate step, not part of this migration.

Known garbage in the source file (see prior analysis): "Google" has ~17
duplicate entries from repeated test runs, and "None"/"Not mentioned" exist
as fake company keys from the unguarded parse bug. This script:
  - migrates every real company's notes into structured_memory + vector_memory
  - routes "None"/"Not mentioned"-style keys into
    structured_memory's unresolved_company_notes table instead of creating
    fake company history in the new stores too
  - does NOT deduplicate the repeated "Google" test-run entries -- that's a
    judgment call left to you (see --dedupe flag) rather than something
    silently done for you
"""
import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
load_dotenv()

import structured_memory
import vector_memory

OLD_MEMORY_FILE = "company_memory.json"

_NOTE_PATTERN = re.compile(
    r"Fit score ([\d.]+),\s*status:\s*([a-zA-Z_]+)"
)


def _parse_note(note: str):
    """Best-effort extraction of fit_score/status from the note text, since
    the old store only ever recorded a free-text string. Returns
    (fit_score, status), either of which may be None if the note doesn't
    match the expected shape (e.g. the free-text TestCorp-style notes)."""
    match = _NOTE_PATTERN.search(note)
    if not match:
        return None, None
    return float(match.group(1)), match.group(2)


def migrate(dedupe: bool = False, dry_run: bool = False):
    if not os.path.exists(OLD_MEMORY_FILE):
        print(f"No {OLD_MEMORY_FILE} found -- nothing to migrate.")
        return

    with open(OLD_MEMORY_FILE, "r") as f:
        data = json.load(f)

    migrated = 0
    unresolved = 0
    skipped_dupes = 0

    for company, notes in data.items():
        is_unresolved = structured_memory._is_unresolved_company(company)

        seen_notes = set()
        for note in notes:
            if dedupe and not is_unresolved:
                if note in seen_notes:
                    skipped_dupes += 1
                    continue
                seen_notes.add(note)

            fit_score, status = _parse_note(note)

            if dry_run:
                dest = "unresolved_company_notes" if is_unresolved else "company_notes"
                print(f"[DRY RUN] -> {dest}: company={company!r} note={note!r}")
                continue

            row_id = structured_memory.save_company_memory(
                company=company, note=note, fit_score=fit_score, status=status
            )

            if is_unresolved:
                unresolved += 1
            else:
                vector_memory.add_memory(
                    company=company, note=note, fit_score=fit_score,
                    status=status, sql_row_id=row_id,
                )
                migrated += 1

    print(f"\nMigration {'(dry run) ' if dry_run else ''}complete:")
    print(f"  {migrated} notes migrated to company_notes + Chroma")
    print(f"  {unresolved} notes routed to unresolved_company_notes (None/Not mentioned/etc.)")
    if dedupe:
        print(f"  {skipped_dupes} exact-duplicate notes skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Skip exact-duplicate notes within the same company (e.g. repeated test runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be migrated without writing to SQLite/Chroma",
    )
    args = parser.parse_args()

    if not args.dry_run:
        confirm = input(
            f"This will write into {structured_memory.DB_FILE} and "
            f"{vector_memory.PERSIST_DIR}/ based on {OLD_MEMORY_FILE}. Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    migrate(dedupe=args.dedupe, dry_run=args.dry_run)
