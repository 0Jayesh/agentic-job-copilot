import __init__
from dotenv import load_dotenv
load_dotenv()

from graph_builder import app

def run_approval_test(thread_id, approved_value):
    config = {"configurable": {"thread_id": thread_id}}

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

    app.invoke(sample_input, config=config)

    # Simulate the human's real decision being recorded
    app.update_state(config, {"approved": approved_value})

    result = app.invoke(None, config=config)
    print(f"\n[thread={thread_id}] approved set to {approved_value} -> final state approved: {result.get('approved')}, status: {result.get('status')}")
    return result

print("=== TEST: Human approves ===")
run_approval_test("approval-test-yes", True)

print("\n=== TEST: Human rejects ===")
run_approval_test("approval-test-no", False)