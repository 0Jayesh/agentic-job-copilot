"""
Minimal local calendar store for check_calendar_conflict.

Deliberately mirrors memory.py's flat-JSON-file pattern rather than
introducing a new storage mechanism — there's no real calendar integration
(Google Calendar API etc.) in scope yet, so this is a local stand-in that
the tool logic can later be pointed at a real calendar without changing its
interface.
"""
import json
import os
from datetime import datetime, timedelta

CALENDAR_FILE = "scheduled_interviews.json"


def _load_all() -> dict:
    if not os.path.exists(CALENDAR_FILE):
        return {}
    with open(CALENDAR_FILE, "r") as f:
        return json.load(f)


def _save_all(data: dict):
    with open(CALENDAR_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_events_on(date_str: str) -> list:
    data = _load_all()
    return data.get(date_str, [])


def add_event(date_str: str, label: str):
    data = _load_all()
    data.setdefault(date_str, [])
    data[date_str].append(label)
    _save_all(data)


def suggest_alternatives(date_str: str, count: int = 3, horizon_days: int = 30) -> list:
    """Walk forward from the requested date and return the next `count` dates
    with no existing entries. Simple and deterministic — no notion of
    business days / working hours, since none of that is modeled yet."""
    data = _load_all()
    base = datetime.fromisoformat(date_str)
    alternatives = []
    offset = 1
    while len(alternatives) < count and offset <= horizon_days:
        candidate = (base + timedelta(days=offset)).date().isoformat()
        if not data.get(candidate):
            alternatives.append(candidate)
        offset += 1
    return alternatives
