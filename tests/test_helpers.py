import __init__ 
from dotenv import load_dotenv
load_dotenv()

from nodes import extract_required_years, extract_candidate_years, years_match

print("--- extract_required_years ---")
print(extract_required_years("4+ years experience"))       # expect (4, None)
print(extract_required_years("4-7 years experience"))      # expect (4, 7)
print(extract_required_years("no years mentioned here"))   # expect (None, None)

print("\n--- extract_candidate_years ---")
print(extract_candidate_years("5+ years of experience"))   # expect 5.0
print(extract_candidate_years("3.5 years of experience"))  # expect 3.5
print(extract_candidate_years("no years mentioned"))       # expect None

print("\n--- years_match ---")
print(years_match(4, None, 5.0))   # expect True  (5 >= 4)
print(years_match(4, None, 3.0))   # expect False (3 < 4)
print(years_match(4, 7, 5.0))      # expect True  (4 <= 5 <= 7)
print(years_match(4, 7, 8.0))      # expect False (8 > 7)
print(years_match(None, None, 5.0)) # expect False (no requirement to check against)

from nodes import semantic_match

print("\n--- semantic_match ---")
print("\n--- semantic_match ---")
print(semantic_match("Bachelor's degree in Computer Science", "Bachelor of Engineering, Computer Science and Engineering"))  # expect True
print(semantic_match("PhD in Astrophysics", "Bachelor of Engineering, Computer Science and Engineering"))  # expect False