# from dotenv import load_dotenv
# load_dotenv()

# from nodes import parse_node, score_node
# import resume_perfect_match
# import resume

# # Temporarily swap in the "perfect match" resume for this test
# resume.RESUME_TEXT = resume_perfect_match.RESUME_TEXT

# sample_input = {
#     "raw_input": "Role: Software Development Engineer. Key Skills: Sklearn, PyTorch, Tensorflow, Spark. 4+ years experience required.",
#     "company": "Test Company",
#     "role": None,
#     "requirements": None,
#     "fit_score": None,
#     "status": "pending"
# }

# result = parse_node(sample_input)
# result = score_node(result)
# print("Fit score:", result["fit_score"])

import __init__ 
from dotenv import load_dotenv
load_dotenv()

from nodes import parse_node, score_node
from tests.resume_perfect_match import RESUME_TEXT as PERFECT_RESUME

sample_input = {
    "raw_input": "Role: Software Development Engineer. Key Skills: Sklearn, PyTorch, Tensorflow, Spark. 4+ years experience required.",
    "company": "Test Company",
    "role": None,
    "requirements": None,
    "fit_score": None,
    "status": "pending"
}

result = parse_node(sample_input)
result = score_node(result, PERFECT_RESUME)
print("Fit score:", result["fit_score"])
print("Matched skills:", result["matched_skills"])
print("Missing skills:", result["missing_skills"])