import json
import os

MEMORY_FILE = "company_memory.json"

def _load_all_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def _save_all_memory(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def save_company_memory(company: str, note: str):
    if not company:
        return
    data = _load_all_memory()
    data.setdefault(company, [])
    data[company].append(note)
    _save_all_memory(data)

def get_company_memory(company: str) -> list:
    if not company:
        return []
    data = _load_all_memory()
    return data.get(company, [])