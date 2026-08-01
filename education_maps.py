import re

# DEGREE_LEVELS = {
#     "BACHELORS": [
#         "BTECH", "B.TECH", "BE", "B.E", 
#         "BSC", "B.SC", "BCA", "B.C.A", 
#         "BBA", "B.B.A", "BCOM", "B.COM", 
#         "BA", "B.A", "BHM", "BACHELOR OF HOTEL MANAGEMENT",
#         "BARCH", "B.ARCH", "LLB", "MBBS", "BPHARM"
#     ],
#     "MASTERS": [
#         "MTECH", "M.TECH", "ME", "M.E", 
#         "MSC", "M.SC", "MCA", "M.C.A", 
#         "MBA", "M.B.A", "PGDM", "PG DIPLOMA", 
#         "MCOM", "M.COM", "MA", "M.A", 
#         "MHM", "MASTER OF HOTEL MANAGEMENT",
#         "MARCH", "M.ARCH", "LLM", "MPHARM"
#     ],
#     "DOCTORATE": ["PHD", "DOCTORATE", "MPHIL"]
# }

# DISCIPLINE_MAP = {
#     "COMPUTER_SCIENCE": [
#         "COMPUTER SCIENCE", "CS", "CSE", "INFORMATION TECHNOLOGY", "IT", 
#         "COMPUTER APPLICATIONS", "SOFTWARE ENGINEERING", "DATA SCIENCE", "AI", "ML", "AI/ML"
#     ],
#     "ELECTRONICS": ["ELECTRONICS", "ECE", "EEE", "ETRX", "TELECOMMUNICATION"],
#     "MECHANICAL": ["MECHANICAL", "MECH", "PRODUCTION", "AUTOMOBILE"],
#     "CIVIL": ["CIVIL", "STRUCTURAL", "CONSTRUCTION"],
#     "MANAGEMENT": [
#         "MANAGEMENT", "BUSINESS ADMINISTRATION", "MBA", "BBA", 
#         "MARKETING", "FINANCE", "HR", "OPERATIONS", "SUPPLY CHAIN"
#     ],
#     "HOSPITALITY": [
#         "HOTEL MANAGEMENT", "HOSPITALITY", "CATERING", "CULINARY", 
#         "TRAVEL", "TOURISM", "HOUSEKEEPING", "FRONT OFFICE"
#     ],
#     "COMMERCE": ["COMMERCE", "ACCOUNTING", "FINANCE", "CA", "CS"],
#     "ARTS": ["ARTS", "ENGLISH", "HISTORY", "ECONOMICS", "PSYCHOLOGY", "SOCIOLOGY"]
# }

DEGREE_LEVELS = {
    "BACHELORS": [
        "BTECH", "B.TECH", "BE", "B.E", 
        "BSC", "B.SC", "BCA", "B.C.A", 
        "BBA", "B.B.A", "BCOM", "B.COM", 
        "BA", "B.A", "BHM", "BACHELOR OF HOTEL MANAGEMENT",
        "BARCH", "B.ARCH", "LLB", "MBBS", "BPHARM",
        "BACHELOR", "BACHELORS" # Handles Bachelor, Bachelors, Bachelor's
    ],
    "MASTERS": [
        "MTECH", "M.TECH", "ME", "M.E", 
        "MSC", "M.SC", "MCA", "M.C.A", 
        "MBA", "M.B.A", "PGDM", "PG DIPLOMA", 
        "MCOM", "M.COM", "MA", "M.A", 
        "MHM", "MASTER OF HOTEL MANAGEMENT",
        "MARCH", "M.ARCH", "LLM", "MPHARM",
        "MASTER", "MASTERS" # Handles Master, Masters, Master's
    ],
    "DOCTORATE": [
        "PHD", "DOCTORATE", "MPHIL", 
        "DOCTOR", "DOCTORS" # Handles Doctor, Doctors, Doctor's
    ]
}

DISCIPLINE_MAP = {
    "COMPUTER_SCIENCE": [
        "COMPUTER SCIENCE",
        "COMPUTER SCIENCE AND ENGINEERING",
        "COMPUTER ENGINEERING",
        "CSE",
        "CS",
        "INFORMATION TECHNOLOGY",
        "COMPUTER APPLICATION",
        "COMPUTER APPLICATIONS",
        "SOFTWARE ENGINEERING",
        "DATA SCIENCE",
        "AI",
        "ML",
        "AI/ML",
        "INFORMATION SCIENCE",
        "INFORMATION SCIENCE AND ENGINEERING",
        "ISE"
    ],
    "ELECTRONICS": [
        "ELECTRONICS", "ECE", "EEE", "ETRX", "TELECOMMUNICATION", "TELECOMMUNICATIONS"
    ],
    "MECHANICAL": [
        "MECHANICAL", "MECH", "PRODUCTION", "AUTOMOBILE"
    ],
    "CIVIL": [
        "CIVIL", "STRUCTURAL", "CONSTRUCTION"
    ],
    "MANAGEMENT": [
        "MANAGEMENT", "BUSINESS ADMINISTRATION", "MBA", "BBA", 
        "MARKETING", "FINANCE", "HR", "OPERATIONS", "SUPPLY CHAIN"
    ],
    "HOSPITALITY": [
        "HOTEL MANAGEMENT", "HOSPITALITY", "CATERING", "CULINARY", 
        "TRAVEL", "TOURISM", "HOUSEKEEPING", "FRONT OFFICE"
    ],
    "COMMERCE": [
        "COMMERCE", "ACCOUNTING", "FINANCE", "CA", "CS"
    ],
    "ARTS": [
        "ARTS", "ENGLISH", "HISTORY", "ECONOMICS", "PSYCHOLOGY", "SOCIOLOGY"
    ]
}


def extract_education_tokens(text: str) -> dict:
    """Scans text to isolate and extract education qualification keys."""
    if not text:
        return {"levels": set(), "disciplines": set()}
        
    clean_text = re.sub(r'[^\w\s]', '', text.upper())
    found_levels = set()
    found_disciplines = set()
    
    for level_key, variations in DEGREE_LEVELS.items():
        for var in variations:
            clean_var = re.sub(r'[^\w\s]', '', var.upper())
            if re.search(r'\b' + re.escape(clean_var) + r'\b', clean_text):
                found_levels.add(level_key)
                
    for discipline_key, variations in DISCIPLINE_MAP.items():
        for var in variations:
            clean_var = re.sub(r'[^\w\s]', '', var.upper())
            if re.search(r'\b' + re.escape(clean_var) + r'\b', clean_text):
                found_disciplines.add(discipline_key)
                
    return {
        "levels": found_levels,
        "disciplines": found_disciplines
    }