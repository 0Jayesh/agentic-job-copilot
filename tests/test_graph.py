import __init__
from dotenv import load_dotenv
load_dotenv()

from graph_builder import app

config = {"configurable": {"thread_id": "test-session-1"}}

print("=== TEST 1: High-fit input (should pause before human_review) ===")
sample_input = {
    "raw_input": "Hi, we'd like to invite you for an interview for the Software Engineer role at Google.",
    "company": None,
    "role": None,
    "fit_score": None,
    "status": "pending",
    "requirements": None,
    "education_required": None,
    "experience_required": None,
    "years_of_experience_required": None
}

result = app.invoke(sample_input, config=config)
print("\nPaused after first invoke. Status:", result.get("status"))
print("Draft present:", "draft_reply" in result)
print("(Graph should have STOPPED here, before human_review actually ran)")

print("\n=== Simulating human approval, resuming graph ===")
resumed_result = app.invoke(None, config=config)
print("\nFinal status after resume:", resumed_result.get("status"))
print("Full final state:", resumed_result)


print("\n\n=== TEST 2: Low-fit input (should skip straight to finalize, no pause) ===")
config2 = {"configurable": {"thread_id": "test-session-2"}}
low_fit_input = {
    "raw_input": "We need a Senior Marine Biologist with 15 years of deep-sea coral reef research experience.",
    "company": None,
    "role": None,
    "fit_score": None,
    "status": "pending",
    "requirements": None,
    "education_required": None,
    "experience_required": None,
    "years_of_experience_required": None
}
result2 = app.invoke(low_fit_input, config=config2)
print("\nFinal status:", result2.get("status"))
print("draft_reply present:", "draft_reply" in result2)