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