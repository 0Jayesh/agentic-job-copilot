import __init__
from dotenv import load_dotenv
load_dotenv()

from nodes import planner_node, drafter_node, semantic_match,  finalize_node

def check(label, actual, expected):
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] {label} -> got {actual}, expected {expected}")

# planner_node
check("planner high score", planner_node({"fit_score": 75.0})["needs_followup"], True)
check("planner low score", planner_node({"fit_score": 45.0})["needs_followup"], False)
check("planner missing score", planner_node({"fit_score": None})["needs_followup"], False)

print("\n[INFO] semantic_match sanity check:", round(semantic_match("Bachelor's degree in Computer Science", "Bachelor of Engineering, Computer Science and Engineering"), 2))

print("\n[INFO] drafter_node sanity check (first 100 chars):")
draft_result = drafter_node({"company": "Unify Technologies", "role": "Software Development Engineer"})
print(draft_result["draft_reply"][:100], "...")

print("\n--- planner_node boundary tests ---")
check("planner exactly 60", planner_node({"fit_score": 60.0})["needs_followup"], True)
check("planner just below 60", planner_node({"fit_score": 59.9})["needs_followup"], False)

print("\n--- drafter_node missing company/role ---")
result = drafter_node({})
print("draft_reply exists:", "draft_reply" in result)
print(result["draft_reply"][:150])

print("\n--- finalize_node ---")
check("finalize approved", finalize_node({"approved": True, "needs_followup": True})["status"], "approved_ready_to_send")
check("finalize rejected", finalize_node({"approved": False, "needs_followup": True})["status"], "rejected_discarded")
check("finalize no followup needed", finalize_node({"approved": None, "needs_followup": False})["status"], "no_action_needed")