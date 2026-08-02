"""
MCP server exposing the job-search-copilot tools.

Design constraints this file follows (per project spec section 3.2):
  1. Every tool's input/output is a Pydantic model, not a loose dict.
  2. Every tool call is individually try/except-wrapped — one tool failing
     never discards results from other tools that succeeded (partial-failure
     handling).
  3. Sequential vs. parallel tool calls are demonstrated explicitly, not
     just described:
       - `process_application` is SEQUENTIAL by necessity: scoring needs
         parse's output, and the calendar check needs a date that may only
         be known after parsing. Each step is awaited before the next runs.
       - `score_against_resumes` is PARALLEL by necessity: scoring the same
         job against N independent resume versions has no cross-dependency,
         so the calls run concurrently via asyncio.gather.

Nothing here modifies nodes.py, state.py, memory.py, education_maps.py,
resume.py, or graph_builder.py — this file only imports and wraps the
existing, already-tested functions.
"""
import asyncio
from typing import Optional, List

from mcp.server.mcpserver import MCPServer

from nodes import (
    parse_node,
    drafter_node,
    check_skills_fit,
    check_years_fit,
    check_education_fit,
    parse_resume_into_sections,
)
from resume import RESUME_TEXT
from resume_document import parse_resume_document as _parse_resume_document
import calendar_store

from models import (
    ParseEmailInput,
    ParseEmailOutput,
    MatchResumeInput,
    FitScoreResult,
    DraftReplyInput,
    DraftReplyOutput,
    ParseResumeDocumentInput,
    ParseResumeDocumentOutput,
    CheckCalendarConflictInput,
    CheckCalendarConflictOutput,
    ProcessApplicationInput,
    ProcessApplicationOutput,
    ScoreAgainstResumesInput,
    ScoreAgainstResumesOutput,
)

server = MCPServer("job-search-copilot-tools")


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools) — shared scoring logic used by
# both match_resume and the orchestration tools below, so the weighting
# logic lives in exactly one place.
# ---------------------------------------------------------------------------

def _score_against_resume(parsed: ParseEmailOutput, resume_text: str) -> FitScoreResult:
    state = parsed.model_dump()

    skill_score, matched, missing = check_skills_fit(state, resume_text)
    years_score = check_years_fit(state, resume_text, matched, missing)
    sections = parse_resume_into_sections(resume_text)
    edu_score, edu_match = check_education_fit(state, sections)

    final_score = round(
        (skill_score * 0.40) + (years_score * 0.30) + (edu_score * 0.15),
        1,
    )

    # Confidence reflects how much real signal parse_email actually gave us.
    # Drops when company/role/requirements came back empty — used for
    # confidence-based HITL routing (spec 3.6), not a stand-in for the score
    # itself.
    signal_fields = [parsed.company, parsed.role, bool(parsed.requirements)]
    confidence = round(sum(1 for f in signal_fields if f) / len(signal_fields), 2)

    reasoning = (
        f"Skills matched: {len(matched)}/{len(matched) + len(missing)}. "
        f"Education match: {edu_match}. "
        f"Weighted score = {skill_score:.1f}*0.40 + {years_score:.1f}*0.30 + {edu_score:.1f}*0.15."
    )

    return FitScoreResult(
        score=final_score,
        reasoning=reasoning,
        matched_skills=matched,
        missing_skills=missing,
        confidence=confidence,
        status="scored",
        error=None,
    )


# ---------------------------------------------------------------------------
# Core tools (existing behavior, now Pydantic-typed)
# ---------------------------------------------------------------------------

@server.tool()
def parse_email(input: ParseEmailInput) -> ParseEmailOutput:
    """Extract company, role, requirements, education, experience, and
    years-of-experience from a raw email or job posting."""
    try:
        state = {"raw_input": input.raw_text}
        result = parse_node(state)
        return ParseEmailOutput(
            company=result.get("company"),
            role=result.get("role"),
            requirements=result.get("requirements") or [],
            education_required=result.get("education_required"),
            experience_required=result.get("experience_required"),
            years_of_experience_required=result.get("years_of_experience_required"),
            status=result.get("status", "generation_failed"),
            error=None,
        )
    except Exception as e:
        return ParseEmailOutput(status="generation_failed", error=str(e))


@server.tool()
def match_resume(input: MatchResumeInput) -> FitScoreResult:
    """Score a parsed job's fit against the stored resume (skills, years, education)."""
    try:
        return _score_against_resume(input.parsed_state, RESUME_TEXT)
    except Exception as e:
        return FitScoreResult(
            score=0.0, reasoning="Scoring failed", confidence=0.0,
            status="generation_failed", error=str(e),
        )


@server.tool()
def draft_reply(input: DraftReplyInput) -> DraftReplyOutput:
    """Generate a follow-up email draft for a given company/role."""
    try:
        state = {"company": input.company, "role": input.role}
        result = drafter_node(state)
        draft = result.get("draft_reply")
        if draft is None:
            return DraftReplyOutput(status="generation_failed", error="LLM returned no draft")
        return DraftReplyOutput(draft=draft, status="ok", error=None)
    except Exception as e:
        return DraftReplyOutput(status="generation_failed", error=str(e))


# ---------------------------------------------------------------------------
# New tools (spec 3.2)
# ---------------------------------------------------------------------------

@server.tool()
def parse_resume_document(input: ParseResumeDocumentInput) -> ParseResumeDocumentOutput:
    """OCR/document-understanding on an uploaded PDF resume — extracts clean
    text before it reaches the scoring step. Real use of multimodal input
    handling: most resumes arrive as PDFs, not plain text."""
    try:
        result = _parse_resume_document(input.file_path)
        return ParseResumeDocumentOutput(**result)
    except Exception as e:
        return ParseResumeDocumentOutput(source="none", status="generation_failed", error=str(e))


@server.tool()
def check_calendar_conflict(input: CheckCalendarConflictInput) -> CheckCalendarConflictOutput:
    """Flag scheduling conflicts with existing interviews on a given date.
    If a conflict exists, suggests up to 3 alternative open dates rather
    than just returning yes/no."""
    try:
        existing = calendar_store.get_events_on(input.date)
        has_conflict = len(existing) > 0

        alternatives = []
        if has_conflict:
            alternatives = calendar_store.suggest_alternatives(input.date, count=3)
        elif input.label:
            calendar_store.add_event(input.date, input.label)

        return CheckCalendarConflictOutput(
            conflict=has_conflict,
            conflicting_events=existing,
            suggested_alternatives=alternatives,
            status="ok",
            error=None,
        )
    except Exception as e:
        return CheckCalendarConflictOutput(
            conflict=False, status="generation_failed", error=str(e),
        )


# ---------------------------------------------------------------------------
# Orchestration tools — explicit sequential vs. parallel demonstration
# ---------------------------------------------------------------------------

@server.tool()
def process_application(input: ProcessApplicationInput) -> ProcessApplicationOutput:
    """SEQUENTIAL pipeline: parse -> score -> (optional) calendar check.

    This must run sequentially, not in parallel: match_resume needs
    parse_email's structured output, and check_calendar_conflict needs the
    interview date, which is only meaningful after parsing has happened.
    Each step is independently try/except-wrapped: a calendar-check failure
    does not discard a successful parse+score result.
    """
    errors: List[str] = []
    parsed_out: Optional[ParseEmailOutput] = None
    fit_out: Optional[FitScoreResult] = None
    calendar_out: Optional[CheckCalendarConflictOutput] = None

    # Step 1: parse (must happen first — everything else depends on it)
    try:
        parsed_out = parse_email(ParseEmailInput(raw_text=input.raw_text))
        if parsed_out.status == "generation_failed":
            errors.append(f"parse failed: {parsed_out.error}")
    except Exception as e:
        errors.append(f"parse raised: {e}")

    # Step 2: score (depends on step 1's output — genuinely sequential)
    if parsed_out and parsed_out.status != "generation_failed":
        try:
            fit_out = match_resume(MatchResumeInput(parsed_state=parsed_out))
        except Exception as e:
            errors.append(f"score raised: {e}")

    # Step 3: calendar check (depends on a date, which we only have after
    # parsing — and is entirely optional, so its failure never blocks 1-2)
    if input.interview_date:
        try:
            calendar_out = check_calendar_conflict(
                CheckCalendarConflictInput(
                    date=input.interview_date,
                    label=f"{parsed_out.company if parsed_out else 'Unknown'} interview",
                )
            )
        except Exception as e:
            errors.append(f"calendar check raised: {e}")

    return ProcessApplicationOutput(
        parsed=parsed_out, fit=fit_out, calendar=calendar_out, errors=errors,
    )


async def _score_one_resume(parsed: ParseEmailOutput, resume_path: str) -> FitScoreResult:
    """Runs the (sync, CPU/IO-bound) scoring function in a thread so multiple
    calls can genuinely overlap under asyncio.gather rather than blocking
    each other."""
    def _read_and_score():
        with open(resume_path, "r") as f:
            resume_text = f.read()
        return _score_against_resume(parsed, resume_text)

    return await asyncio.to_thread(_read_and_score)


@server.tool()
def score_against_resumes(input: ScoreAgainstResumesInput) -> ScoreAgainstResumesOutput:
    """PARALLEL scoring: the same parsed job scored against N independent,
    plain-text resume files at once. None of these calls depend on each
    other's results, so they run concurrently via asyncio.gather rather than
    one after another.
    """

    async def _run_all():
        tasks = [
            _score_one_resume(input.parsed_state, path) for path in input.resume_paths
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    raw_results = asyncio.run(_run_all())

    results: List[FitScoreResult] = []
    errors: List[str] = []
    for path, r in zip(input.resume_paths, raw_results):
        if isinstance(r, Exception):
            errors.append(f"{path}: {r}")
        else:
            results.append(r)

    return ScoreAgainstResumesOutput(results=results, errors=errors)
