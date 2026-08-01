from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    raw_input: str
    company: Optional[str]
    role: Optional[str]
    # requirements: Optional[str]
    requirements: Optional[List[str]]
    education_required: Optional[str]
    experience_required: Optional[str]
    years_of_experience_required: Optional[str]

    fit_score: Optional[float]
    matched_skills: Optional[List[str]]
    missing_skills: Optional[List[str]]
    status: str
    # New additions for this step - separate category matching
    education_match: Optional[bool]
    experience_match: Optional[bool]
    needs_followup: Optional[bool]
    draft_reply: Optional[str]
    approved: Optional[bool]