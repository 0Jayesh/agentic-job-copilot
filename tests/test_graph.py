import __init__ 
from dotenv import load_dotenv
load_dotenv()

from graph_builder import app

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

result = app.invoke(sample_input)
print(result)

print("\n\n=== TEST 2: Low-fit input (should skip draft) ===")
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
result2 = app.invoke(low_fit_input)
print("\nFinal needs_followup:", result2.get("needs_followup"))
print("draft_reply present:", "draft_reply" in result2)