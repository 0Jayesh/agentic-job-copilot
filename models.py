"""
Pydantic models for every MCP tool's input/output.

Why this file exists: the original mcp_server.py passed loose dicts in and
out of every tool. That's the exact fragility pattern called out in the
project spec (section 3.2) — a malformed/partial LLM output silently
propagates downstream instead of failing validation immediately. Every tool
in mcp_server.py now takes a typed Pydantic input and returns a typed
Pydantic output.

Nothing in nodes.py, state.py, memory.py, or graph_builder.py is touched or
imported here beyond field-shape reuse — this is a pure typing layer that
sits in front of the existing, tested node functions.
"""
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ---- parse_email ----

class ParseEmailInput(BaseModel):
    raw_text: str = Field(..., description="Raw recruiter email or job posting text")


class ParseEmailOutput(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    requirements: List[str] = Field(default_factory=list)
    education_required: Optional[str] = None
    experience_required: Optional[str] = None
    years_of_experience_required: Optional[str] = None
    status: str
    error: Optional[str] = None


# ---- match_resume ----

class MatchResumeInput(BaseModel):
    parsed_state: ParseEmailOutput = Field(
        ..., description="Output of parse_email — the dependency this tool needs"
    )


class FitScoreResult(BaseModel):
    """Structured scoring result. `score` is the deterministic weighted
    number from the existing scoring pipeline (skills/years/education/exp
    weighted average) — NOT an LLM-guessed number. `confidence` reflects how
    much of the underlying signal was actually available (e.g. drops when
    company/role/requirements came back empty), used for confidence-based
    HITL routing per spec 3.6."""

    score: float
    reasoning: str
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    confidence: float
    status: str
    error: Optional[str] = None


# ---- draft_reply ----

class DraftReplyInput(BaseModel):
    company: str
    role: str


class DraftReplyOutput(BaseModel):
    draft: Optional[str] = None
    status: str
    error: Optional[str] = None


# ---- parse_resume_document (new — OCR / document understanding) ----

class ParseResumeDocumentInput(BaseModel):
    file_path: str = Field(..., description="Path to a PDF resume on disk")


class ParseResumeDocumentOutput(BaseModel):
    extracted_text: Optional[str] = None
    source: str  # "gemini_document_understanding" | "ocr_fallback" | "none"
    status: str
    error: Optional[str] = None


# ---- check_calendar_conflict (new) ----

class CheckCalendarConflictInput(BaseModel):
    date: str = Field(..., description="ISO date string, e.g. 2026-08-10")
    label: Optional[str] = Field(
        None, description="What to schedule on this date if there's no conflict"
    )


class CheckCalendarConflictOutput(BaseModel):
    conflict: bool
    conflicting_events: List[str] = Field(default_factory=list)
    suggested_alternatives: List[str] = Field(default_factory=list)
    status: str
    error: Optional[str] = None


# ---- orchestration tools (sequential / parallel demos) ----

class ProcessApplicationInput(BaseModel):
    raw_text: str
    interview_date: Optional[str] = Field(
        None, description="If the email mentions an interview date, pass it as ISO date"
    )


class ProcessApplicationOutput(BaseModel):
    """Result of the sequential pipeline: parse -> score -> (optional) calendar check.
    Each step's failure is isolated — a calendar-check failure does not
    discard successful parse/score results."""

    parsed: Optional[ParseEmailOutput] = None
    fit: Optional[FitScoreResult] = None
    calendar: Optional[CheckCalendarConflictOutput] = None
    errors: List[str] = Field(default_factory=list)


class ScoreAgainstResumesInput(BaseModel):
    parsed_state: ParseEmailOutput
    resume_paths: List[str] = Field(
        ..., description="Multiple stored resume versions to score independently in parallel"
    )


class ScoreAgainstResumesOutput(BaseModel):
    results: List[FitScoreResult]
    errors: List[str] = Field(default_factory=list)
