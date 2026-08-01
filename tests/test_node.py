import __init__ 
from dotenv import load_dotenv
load_dotenv()

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import parse_node, score_node
from resume import RESUME_TEXT
# from resume_perfect_match import RESUME_TEXT

# JD
sample_input = {
    "raw_input": """Dear Candidate,
I hope you are doing well; I'm reaching out from Unify Technologies - Hyderabad.
We have followed the Job Opportunity at our company, please go through the job details and let us know if you are interested.
Our Company: Unify Technologies
Company Nature of work: IT-Software Product Development and Product Engineering Services
Founded Year: 2015
Company Locations: USA, Hyderabad, Bangalore, Pune, Chandigarh, Gurgaon
Num of Total Employees: 1,500+
Employment Type: Full-Time
Role: Software Development Engineer
Position: Apple- Maps
Experience Required: 4+ Years
Key Skills: Sklearn, PyTorch, Tensorflow, Spark
Job Location: Hyderabad, India (Hybrid – Work from Office)
Responsibilities
Data Pipeline Management
Maintain and optimize training data pipelines
Model Lifecycle Management
Onboard new ML models onto the existing platform
MLOps
Maintain and enhance CI/CD pipelines for ML data & model pipelines
Key Qualifications
Bachelor's degree in Computer Science, Engineering, or a related field.
Minimum 4+ years of experience in software engineering
Prior hands on experience in ML model productionalization and operationalization
Familiarity with ML frameworks such as Sklearn, PyTorch, Tensorflow
Familiarity with data processing frameworks such as Spark""",
    "company": None,
    "role": None,
    "requirements": None,
    "fit_score": None,
    "status": "pending"
}

# result = parse_node(sample_input)
# print("Parsed requirements:", result["requirements"])

result = parse_node(sample_input)
# print("F",result)
print("\nCOMPANY:", result["company"] )
print("\nROLE:", result["role"] )
print("\nREQUIREMENTS:", result["requirements"] )
print("\nEDUCATION:", result["education_required"] )
print("\nEXPERIENCE DESCIPTION:", result["experience_required"] )
print("\nYearsOfExperience:", result["years_of_experience_required"] )

result = score_node(result, RESUME_TEXT)
print("\nFit score:", result["fit_score"])
# print("Matched skills:", result["matched_skills"])
# print("Missing skills:", result["missing_skills"])