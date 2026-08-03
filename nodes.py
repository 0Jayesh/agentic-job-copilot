from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import re
from education_maps import extract_education_tokens
from structured_memory import save_company_memory, get_company_memory, is_unresolved_company

# llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

from langchain_groq import ChatGroq
import time

llm_fallback = ChatGroq(model="llama-3.3-70b-versatile")

def invoke_with_fallback(prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n[DEBUG] Gemini call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print("\n[DEBUG] Gemini exhausted retries. Falling back to Groq.")
    try:
        response = llm_fallback.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"\n[DEBUG] Groq fallback also failed: {e}")
        return None

def parse_llm_response_text(text: str, state: AgentState) -> AgentState:
    """Extracts fields from raw LLM text, robust to minor formatting differences
    (markdown bold, capitalization, leading symbols) since different providers
    (Gemini vs Groq) may format their output slightly differently."""

    def clean_line(line: str) -> str:
        return line.strip().lstrip("*#-").strip()

    for raw_line in text.split("\n"):
        line = clean_line(raw_line)
        line_lower = line.lower()

        if line_lower.startswith("company:"):
            state["company"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("role:"):
            state["role"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("requirements:"):
            raw_reqs = line.split(":", 1)[1].strip()
            state["requirements"] = [r.strip() for r in raw_reqs.split(",") if r.strip() and r.strip().lower() != "none"]
        elif line_lower.startswith("education:"):
            state["education_required"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("experience:"):
            state["experience_required"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("years of experience:"):
            state["years_of_experience_required"] = line.split(":", 1)[1].strip()

    # Safety check: if we got NOTHING usable, this is a parse failure, not "no info in JD"
    if not state.get("company") and not state.get("role"):
        state["status"] = "generation_failed"
        print("\n[DEBUG] Parse produced no usable fields - flagging as generation_failed, not a false empty-JD pass.")
        return state

    state["status"] = "parsed"
    return state


def parse_node(state: AgentState) -> AgentState:
    prompt = f"""Extract information from this text into exactly these following categories:
Company: <name>
Role: <title>
Requirements: <comma-separated list of TOOL/LIBRARY/SKILLS/TechnologyName names only. Don't limit the search in any section, search everywhere in JD
Return only clean keywords or tool names, NOT full descriptions or long phrases. Example Good: PyTorch, Docker, MLOps, Spark, Data Pipelines. 
Example Bad: Maintain and optimize training data pipelines, Prior hands on experience in ML model.>
Education: <the education requirement described in one short phrase, or "None" if not mentioned>
Experience: <any descriptive experience requirement (not years, not tools) in one short phrase, or "None" if not mentioned>
Years of Experience: <specific years of experience required for the role - either in range ( eg: 4 - 7 ), or minimum number ( eg: 4+ ).
Output should be either in format : a-b OR a+ ( 0 for freshers if not found any such relevant information )>

Text: {state['raw_input']}"""

    text = invoke_with_fallback(prompt)
    print(f"\n[DEBUG] RAW TEXT FROM LLM:\n---\n{text}\n---")
    if text is None:
        state["status"] = "generation_failed"
        print("\n[DEBUG] Parser FAILED - both providers down. Halting with error status.")
        return state

    return parse_llm_response_text(text, state)

# def parse_node(state: AgentState) -> AgentState:
#     prompt = f"""Extract information from this text into exactly these following categories:
# Company: <name>
# Role: <title>
# Requirements: <comma-separated list of TOOL/LIBRARY/SKILLS/TechnologyName names only. Don't limit the search in any section, search everywhere in JD
# Return only clean keywords or tool names, NOT full descriptions or long phrases. Example Good: PyTorch, Docker, MLOps, Spark, Data Pipelines. 
# Example Bad: Maintain and optimize training data pipelines, Prior hands on experience in ML model.>
# Education: <the education requirement described in one short phrase, or "None" if not mentioned>
# Experience: <any descriptive experience requirement (not years, not tools) in one short phrase, or "None" if not mentioned>
# Years of Experience: <specific years of experience required for the role - either in range ( eg: 4 - 7 ), or minimum number ( eg: 4+ ).
# Output should be either in format : a-b OR a+ ( 0 for freshers if not found any such relevant information )>

# Text: {state['raw_input']}"""

#     # text = invoke_with_fallback(prompt)
#     text = invoke_with_fallback(prompt)
#     if text is None:
#         state["status"] = "generation_failed"
#         print("\n[DEBUG] Parser FAILED - both providers down. Halting with error status.")
#         return state

#     for line in text.split("\n"):
#         if line.startswith("Company:"):
#             state["company"] = line.replace("Company:", "").strip()
#         if line.startswith("Role:"):
#             state["role"] = line.replace("Role:", "").strip()
#         # if line.startswith("Requirements:"):
#         #     state["requirements"] = line.replace("Requirements:", "").strip()
#         if line.startswith("Requirements:"):
#             raw_reqs = line.replace("Requirements:", "").strip()
#             # state["requirements"] = [r.strip() for r in raw_reqs.split(",") if r.strip()]
#             state["requirements"] = [r.strip() for r in raw_reqs.split(",") if r.strip() and r.strip().lower() != "none"]
#         if line.startswith("Education:"):
#             state["education_required"] = line.replace("Education:", "").strip()
#         if line.startswith("Experience:"):
#             state["experience_required"] = line.replace("Experience:", "").strip()
#         if line.startswith("Years of Experience:"):
#             state["years_of_experience_required"] = line.replace("Years of Experience:", "").strip()

#     state["status"] = "parsed"
#     return state


embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_required_years(years_str: str):
    """
    Parses state["years_of_experience_required"] which looks like "4+" or "4-5".
    Returns (min_years, max_years)
    """
    if not years_str or years_str.strip() == "0":
        return 0, None
        
    # Check for range format like "4-5"
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', years_str)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # Check for minimum plus format like "4+"
    plus_match = re.search(r'(\d+)\s*\+?', years_str)
    if plus_match:
        return int(plus_match.group(1)), None

    return 0, None


def extract_candidate_years(text: str):
    """Returns candidate's stated years as a single float from their resume summary"""
    match = re.search(r'(\d+(?:\.\d+)?)\+?\s*years?', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0  # Default to 0 if not explicitly found


def years_match(required_min, required_max, candidate_years):
    if candidate_years is None:
        candidate_years = 0.0
    if required_min is None:
        return True
        
    if required_max is not None:
        return required_min <= candidate_years <= required_max
    return candidate_years >= required_min


def semantic_match(requirement_text: str, resume_text: str) -> float:
    """Returns the cosine similarity score (0.0 to 1.0) between requirement and resume."""
    if not requirement_text or requirement_text.strip().lower() == "none":
        return 1.0  # Perfect score if nothing is required
        
    req_embedding = embedder.encode([requirement_text])
    resume_embedding = embedder.encode([resume_text])
    similarity = cosine_similarity(req_embedding, resume_embedding)[0][0]
    return max(0.0, float(similarity))


def extract_resume_section(resume_text: str, start_header: str, next_headers: list[str]) -> str:
    """
    Locates the start_header line in the resume, reads lines sequentially, 
    and clips the content immediately when it hits any header in next_headers.
    """
    lines = resume_text.split("\n")
    section_lines = []
    in_section = False

    for line in lines:
        cleaned_line = line.strip()
        
        # 1. Detect the start header (case-insensitive, exact line match)
        if cleaned_line.upper() == start_header.upper():
            in_section = True
            continue # Skip printing the header name itself

        if in_section:
            # 2. If we hit any of the subsequent section headers, stop clipping immediately
            if cleaned_line.upper() in [h.upper() for h in next_headers]:
                break
            section_lines.append(line)

    return "\n".join(section_lines).strip()

def check_education_fit(state: AgentState, sections: dict) -> tuple[float, bool]:
    """Isolates the EDUCATION block and computes a deterministic token match."""
    edu_req = state.get("education_required", "None")
    if not edu_req or edu_req.strip().lower() == "none":
        return 100.0, True

    # 1. Extract requirement tokens from the JD string
    jd_tokens = extract_education_tokens(edu_req)

    # 2. Extract candidate tokens from the isolated resume section (no fallback)
    isolated_edu = sections.get("education", "")

    candidate_tokens = extract_education_tokens(isolated_edu)
    print(f"\n[TOKEN DEBUG] Job Description Needs: {jd_tokens}")
    print(f"[TOKEN DEBUG] Candidate Actually Has: {candidate_tokens}")

    # 3. Intersect the matrices (The P and C Check)
    level_match = bool(jd_tokens["levels"].intersection(candidate_tokens["levels"]))
    discipline_match = bool(jd_tokens["disciplines"].intersection(candidate_tokens["disciplines"]))

    if not jd_tokens["disciplines"]:
        discipline_match = True

    # 4. Final Binary Logic Allocation
    is_match = level_match and discipline_match
    score = 100.0 if is_match else 0.0

    return score, is_match


def check_skills_fit(state: AgentState, resume_text: str):
    resume_lower = resume_text.lower()
    required_skills = [s.strip().lower() for s in (state.get("requirements") or []) if s.strip()]
    skill_only = [s for s in required_skills if "year" not in s]

    matched_skills = [s for s in skill_only if s and s in resume_lower]
    missing_skills = [s for s in skill_only if s and s not in resume_lower]
    skill_score = (len(matched_skills) / len(skill_only) * 100) if skill_only else 100.0

    print(f"\n[DEBUG] Skills Match: {matched_skills} (Score: {skill_score}%)")
    print(f"[DEBUG] Matched Skills Array: {matched_skills}")
    print(f"[DEBUG] Missing Skills Array: {missing_skills}")

    return skill_score, matched_skills, missing_skills


def check_years_fit(state: AgentState, resume_text: str, matched_skills: list, missing_skills: list):
    years_req_string = str(state.get("years_of_experience_required", "0"))
    req_min, req_max = extract_required_years(years_req_string)
    candidate_years = extract_candidate_years(resume_text)

    years_ok = years_match(req_min, req_max, candidate_years)
    print(f"[DEBUG] Experience Check: Required Min: {req_min}, Candidate: {candidate_years} (Match: {years_ok})")

    state["experience_match"] = years_ok

    if years_ok:
        matched_skills.append(f"years_of_experience ({candidate_years} yrs)")
        years_score = 100.0
    else:
        missing_skills.append(f"years_of_experience (needs {years_req_string}, candidate has {candidate_years})")
        if req_min > 0 and candidate_years > 0:
            years_score = min(100.0, (candidate_years / req_min) * 100.0)
        else:
            years_score = 0.0

    print(f"[DEBUG] Experience Ok: {state['experience_match']} | Score: {years_score}%")
    return years_score


def check_experience_description_fit(state: AgentState, sections: dict):
    exp_desc_req = state.get("experience_required") or "None"

    isolated_exp_text = sections.get("experience", "")

    exp_desc_similarity = semantic_match(exp_desc_req, isolated_exp_text)
    exp_desc_score = exp_desc_similarity * 100

    print(f"[DEBUG] Isolated Experience Text:\n{isolated_exp_text}")
    print(f"[DEBUG] Experience Description Similarity: {exp_desc_similarity:.2f} | Score: {exp_desc_score:.1f}%")

    return exp_desc_score

def parse_resume_into_sections(resume_text: str) -> dict:
    CATEGORY_WORDS = {"education", "skill", "experience", "project"}

    def categorize_header(header_line: str) -> str:
        words = header_line.strip().split()
        for word in words:
            clean_word = word.lower()
            if clean_word.endswith("s"):
                clean_word = clean_word[:-1]
            if clean_word in CATEGORY_WORDS:
                return clean_word
        return None

    sections = {}
    current_category = None
    current_lines = []

    lines = resume_text.split("\n")
    for line in lines:
        stripped = line.strip()
        is_header = stripped != "" and stripped == stripped.upper() and any(c.isalpha() for c in stripped)

        if is_header:
            if current_category:
                sections[current_category] = "\n".join(current_lines).strip()
            current_category = categorize_header(stripped)
            current_lines = []
        else:
            if current_category:
                current_lines.append(line)

    if current_category:
        sections[current_category] = "\n".join(current_lines).strip()

    return sections

def score_node(state: AgentState, resume_text: str) -> AgentState:
    sections = parse_resume_into_sections(resume_text)

    skill_score, matched_skills, missing_skills = check_skills_fit(state, resume_text)

    years_score = check_years_fit(state, resume_text, matched_skills, missing_skills)

    edu_score, edu_match = check_education_fit(state, sections)
    state["education_match"] = edu_match
    print(f"\n[DEBUG] Isolated Education Text:\n{sections.get('education', '')}")
    print(f"[DEBUG] Education Match Status: {state['education_match']} | Points Awarded: {edu_score:.1f}%")

    exp_desc_score = check_experience_description_fit(state, sections)

    final_score = round(
        (skill_score * 0.40) +
        (years_score * 0.30) +
        (edu_score * 0.15) +
        (exp_desc_score * 0.15),
        1
    )

    state["fit_score"] = final_score
    state["matched_skills"] = matched_skills
    state["missing_skills"] = missing_skills
    state["status"] = "scored"
    print(f"\n[DEBUG] Pipeline Completed! Final Candidate Fit Score: {state['fit_score']}%")

    return state

def planner_node(state: AgentState) -> AgentState:
    fit_score = state.get("fit_score") or 0.0

    if fit_score >= 60:
        state["needs_followup"] = True
    else:
        state["needs_followup"] = False

    print(f"\n[DEBUG] Planner Decision: fit_score={fit_score} -> needs_followup={state['needs_followup']}")
    return state

def drafter_node(state: AgentState) -> AgentState:
    company = state.get("company", "the company")
    role = state.get("role", "the role")

    prompt = f"""Write a short, professional follow-up email expressing interest in the {role} position at {company}. Keep it under 100 words."""

    # text = invoke_with_fallback(prompt)
    # state["draft_reply"] = text

    text = invoke_with_fallback(prompt)
    if text is None:
        state["draft_reply"] = None
        state["status"] = "generation_failed"
        print("\n[DEBUG] Drafter FAILED - both providers down. Halting with error status.")
    else:
        state["draft_reply"] = text

    print(f"\n[DEBUG] Draft generated:\n{text}")
    return state

def route_after_planning(state: AgentState):
    if state.get("needs_followup"):
        return "draft"
    return "end"

def human_review_node(state: AgentState) -> AgentState:
    print(f"\n[DEBUG] Awaiting human approval for draft:\n{state.get('draft_reply')}")
    return state

def finalize_node(state: AgentState) -> AgentState:
    # Deferred import: vector_memory.py imports `embedder` back from this
    # module. Importing it here (at call time) instead of at the top of the
    # file avoids a circular-import failure that depends on which module
    # happens to import nodes.py first -- by the time finalize_node is
    # actually called, nodes.py is always already fully loaded.
    import vector_memory

    approved = state.get("approved")

    if state.get("needs_followup") is False:
        state["status"] = "no_action_needed"
    elif approved is True:
        state["status"] = "approved_ready_to_send"
    elif approved is False:
        state["status"] = "rejected_discarded"
    else:
        state["status"] = "pending_approval"

    note = f"Fit score {state.get('fit_score')}, status: {state['status']}"
    company = state.get("company")
    row_id = save_company_memory(company, note)

    if not is_unresolved_company(company):
        vector_memory.add_memory(
            company=company,
            note=note,
            fit_score=state.get("fit_score"),
            status=state["status"],
            sql_row_id=row_id,
        )

    print(f"\n[DEBUG] Finalized. Status: {state['status']}")
    return state

def memory_lookup_node(state: AgentState) -> AgentState:
    company = state.get("company")
    past_notes = get_company_memory(company)
    state["past_company_notes"] = past_notes
    print(f"\n[DEBUG] Memory lookup for '{company}': {past_notes}")
    return state