import __init__
from dotenv import load_dotenv
load_dotenv()

from memory import save_company_memory, get_company_memory, _load_all_memory, _save_all_memory

# Clean slate for this test run - remove any leftover TestCorp data from previous runs
all_memory = _load_all_memory()
all_memory.pop("TestCorp", None)
_save_all_memory(all_memory)

print("--- test_memory ---")

print("Before saving:", get_company_memory("TestCorp"))

save_company_memory("TestCorp", "Interviewed March 2026, rejected - lacked Spark experience")
save_company_memory("TestCorp", "Reached out again May 2026, no response")

print("After saving 2 notes:", get_company_memory("TestCorp"))

print("Different company (should be empty):", get_company_memory("OtherCorp"))

from nodes import memory_lookup_node

print("\n--- memory_lookup_node ---")
result = memory_lookup_node({"company": "TestCorp"})
print("past_company_notes:", result["past_company_notes"])

result2 = memory_lookup_node({"company": "BrandNewCorp"})
print("past_company_notes (new company):", result2["past_company_notes"])