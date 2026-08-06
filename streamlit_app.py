"""
streamlit_app.py
Deployment UI (spec section 9 / step 9 of the build order), for HF Spaces
(Streamlit SDK).
Deliberately split into two kinds of tabs:
  - "Score a Job" -- the real thing, calls the actual compiled LangGraph
    (graph_builder.app), including the real HITL interrupt/approve/reject
    flow. This is the only tab that spends API quota.
  - "Company Memory" / "Graph Diagram" -- read-only, zero API cost. Safe
    for any number of visitors to click around in without touching
    Gemini/Groq quota, since a public deployment shares that quota across
    every visitor, not just you.
Uses one thread_id per browser session (st.session_state), so LangGraph's
checkpointer keeps each visitor's in-progress run isolated from everyone
else's -- this is exactly what MemorySaver + thread_id already gives us,
just wired to a UI instead of a test script's hardcoded thread_id.
"""
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass
  
import os
import tempfile
import uuid

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from graph_builder import app as graph_app
import structured_memory
import vector_memory
import graph_viz
import react_from_scratch
import golden_set
import mlflow_setup
import resume_document
import calendar_store

st.set_page_config(page_title="Job Search Copilot", page_icon="🧭", layout="wide")

st.markdown("<h1 style='text-align: center;margin-bottom: 24px'>🧭 Job Search Copilot</h1>", unsafe_allow_html=True)
st.caption(
    "An agentic job-search assistant: parses postings, scores fit against a resume, "
    "drafts replies with human-in-the-loop approval, remembers company history, and "
    "exposes the underlying agent mechanics (MCP tools, ReAct loop, tracing, eval) "
    "for inspection."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

tab_score, tab_memory, tab_graph, tab_advanced, tab_tools = st.tabs(
    ["📄 Score a Job", "🧠 Company Memory", "🕸️ Graph Diagram", "🧪 Advanced", "🛠️ More Tools"]
)


# ===========================================================================
# TAB 1 -- Score a Job (the only tab that spends real API quota)
# ===========================================================================
with tab_score:
    st.header("Score a Job Posting")
    st.caption(
        "Paste a recruiter email or job posting below. This calls the real "
        "agent pipeline (Gemini, with a Groq fallback) -- each analysis "
        "uses a small amount of shared API quota."
    )

    raw_input = st.text_area("Job posting / recruiter email", height=200)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    col_a, col_b = st.columns([1, 1])

    if col_a.button("Analyze", type="primary", disabled=not raw_input.strip()):
        sample_input = {
            "raw_input": raw_input,
            "company": None,
            "role": None,
            "requirements": None,
            "fit_score": None,
            "status": "pending",
            "education_required": None,
            "experience_required": None,
            "years_of_experience_required": None,
        }
        with st.spinner("Parsing, scoring, and checking company history..."):
            result = graph_app.invoke(sample_input, config=config)
        st.session_state.last_result = result
        st.session_state.awaiting_approval = result.get("status") not in (
            None,
            "approved_ready_to_send",
            "rejected_discarded",
            "no_action_needed",
        ) and "draft_reply" in result

    if col_b.button("Start a new session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_result = None
        st.session_state.awaiting_approval = False
        st.rerun()

    result = st.session_state.last_result
    if result:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Company", result.get("company") or "Unresolved")
        c2.metric("Role", result.get("role") or "-")
        c3.metric("Fit score", f"{result.get('fit_score', 0):.1f}%" if result.get("fit_score") is not None else "-")

        with st.expander("Matched / missing skills"):
            st.write("**Matched:**", result.get("matched_skills") or [])
            st.write("**Missing:**", result.get("missing_skills") or [])

        if result.get("past_company_notes"):
            with st.expander(f"Past history with {result.get('company')}"):
                for note in result["past_company_notes"]:
                    st.write("-", note)

        if st.session_state.awaiting_approval and result.get("draft_reply"):
            st.subheader("Draft reply -- your approval needed")
            st.text_area("Draft", value=result["draft_reply"], height=150, disabled=True)
            st.caption(
                "Approving records your decision and finalizes the run -- it does NOT "
                "actually send an email. Copy the draft above if you want to send it yourself."
            )

            approve_col, reject_col = st.columns(2)
            if approve_col.button("Approve", type="primary"):
                graph_app.update_state(config, {"approved": True})
                final = graph_app.invoke(None, config=config)
                st.session_state.last_result = final
                st.session_state.awaiting_approval = False
                st.rerun()
            if reject_col.button("Reject"):
                graph_app.update_state(config, {"approved": False})
                final = graph_app.invoke(None, config=config)
                st.session_state.last_result = final
                st.session_state.awaiting_approval = False
                st.rerun()
        elif result.get("status"):
            st.info(f"Final status: **{result['status']}**")


# ===========================================================================
# TAB 2 -- Company Memory (read-only, zero API cost)
# ===========================================================================
with tab_memory:
    st.header("Company Memory")
    st.caption("Browse past interactions with a company. No API calls -- reads directly from the structured store.")

    company_query = st.text_input("Company name (partial match ok)")
    if company_query:
        notes = structured_memory.get_company_memory_detailed(company_query)
        if notes:
            for row in notes:
                st.write(f"**{row.get('company', '')}** — {row.get('created_at', '')[:10]} -- {row.get('note')}")
        else:
            st.write("No history found for this company yet.")

    st.divider()
    st.subheader("Semantic search across all past notes")
    st.caption(
        "This is embedding-based search (Chroma), not exact text matching -- describe a "
        "situation in your own words and it finds semantically similar past notes, even "
        "across different companies. Still no API calls: the embedding model runs locally."
    )
    semantic_query = st.text_input(
        "e.g. \"companies that rejected me for missing cloud experience\"",
        key="semantic_query",
    )
    if semantic_query:
        similar = vector_memory.query_similar(semantic_query, n_results=5)
        if similar:
            for hit in similar:
                company = hit.get("metadata", {}).get("company", "Unknown")
                st.write(f"**{company}** (distance {hit.get('distance', 0):.3f}) -- {hit.get('note')}")
        else:
            st.write("No notes recorded yet -- run a few analyses in the Score a Job tab first.")

    with st.expander("Companies with a low fit score and no follow-up"):
        low_fit = structured_memory.query_low_fit_no_followup(threshold=60.0)
        if low_fit:
            for row in low_fit:
                st.write(f"**{row['company']}** -- score {row['fit_score']} -- {row['created_at'][:10]}")
        else:
            st.write("None recorded yet.")


# ===========================================================================
# TAB 3 -- Graph Diagram (read-only, zero API cost)
# ===========================================================================
with tab_graph:
    st.header("Agent Graph")
    st.caption("The actual LangGraph structure this app runs, rendered live -- not a hand-drawn architecture picture.")

    diagram_path = graph_viz.save_static_diagram("graph_diagram.png")
    if diagram_path.endswith(".png"):
        st.image(diagram_path)
    else:
        st.write("PNG rendering unavailable in this environment -- raw Mermaid source below (paste into https://mermaid.live):")
        with open(diagram_path) as f:
            st.code(f.read(), language="text")
    st.divider()
    st.subheader("Execution Path for Your Last Run")
    st.caption(
        "The diagram above shows the graph's overall shape -- the same every time. "
        "This shows which nodes actually ran, in order, for your specific last analysis "
        "in the 'Score a Job' tab, reconstructed live from LangGraph's checkpoint history "
        "for your session (thread_id). No API calls -- pure state introspection."
    )

    if st.session_state.get("last_result"):
        execution_path = graph_viz.get_execution_path(st.session_state.thread_id)
        if execution_path:
            st.success(" → ".join(execution_path))
            with st.expander("Raw node sequence"):
                for i, node_name in enumerate(execution_path, start=1):
                    st.write(f"{i}. `{node_name}`")
        else:
            st.info("No checkpoint history found for this session yet.")
    else:
        st.info(
            "Run an analysis in the 'Score a Job' tab first, then come back here to see "
            "exactly which nodes executed for that run -- e.g. whether it stopped after "
            "'finalize' with no follow-up needed, or paused at 'human_review' awaiting your approval."
        )

# ===========================================================================
# TAB 4 -- Advanced: ReAct demo (spends API quota) + capability scorecard
# (zero cost, mostly static text describing what else this project builds
# that has no natural home in a click-through UI)
# ===========================================================================
with tab_advanced:
    st.header("ReAct Agent -- Built From Scratch")
    st.caption(
        "A hand-rolled Thought -> Action -> Observation loop -- no LangGraph, no agent "
        "framework, just raw LLM calls in a while loop, calling the same match_resume "
        "logic as the Score a Job tab. This is a separate, standalone proof of how "
        "agent frameworks work underneath. Uses a small amount of shared API quota."
    )

    react_query = st.text_area(
        "Ask it to evaluate a job posting",
        value=(
            "Score this job against my resume and tell me if I should apply: "
            "'AI Engineer role at Vertex Labs. Required: TensorFlow, Python, "
            "LangChain, RAG. Bachelor's in CS. 3+ years experience.'"
        ),
        height=100,
    )

    if st.button("Run ReAct Demo", type="primary"):
        with st.spinner("Running Thought -> Action -> Observation loop..."):
            react_result = react_from_scratch.run_react_loop_steps(react_query)

        for i, step in enumerate(react_result["steps"], start=1):
            with st.expander(f"Step {i}", expanded=True):
                st.text(step["response_text"])
                if step["observation"]:
                    st.info(f"Observation: {step['observation']}")

        if react_result["final_answer"]:
            st.success(f"Final Answer: {react_result['final_answer']}")
        elif react_result["stopped_reason"]:
            st.warning(f"Stopped: {react_result['stopped_reason']}")

    st.divider()
    st.header("What Else This Project Builds")
    st.caption("Not everything fits a click-through demo -- here's what's running underneath, verified by its own tests.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("MCP Server")
        st.write(
            "5 Pydantic-typed tools exposed via a custom MCP server: `parse_email`, "
            "`match_resume`, `draft_reply`, `parse_resume_document` (PDF/OCR), "
            "`check_calendar_conflict` -- plus 2 orchestration tools demonstrating "
            "sequential vs. parallel tool-calling explicitly (`process_application`, "
            "`score_against_resumes`)."
        )

        st.subheader("Golden-Set Eval + CI")
        st.write(
            f"{len(golden_set.GOLDEN_SET)} hand-written test cases run against the real "
            "pipeline on every push via GitHub Actions -- deliberately includes edge "
            "cases like an unresolved-company input, not just happy paths."
        )

    with col2:
        st.subheader("MLflow Tracing")
        tracing_status = "Enabled" if mlflow_setup._tracing_enabled else "Not yet enabled this session"
        st.write(f"`mlflow.langchain.autolog()` status (live check): **{tracing_status}**")
        st.write("Every node execution and LLM call in this run is traced automatically. View with `mlflow ui`.")

        st.subheader("Memory")
        st.write(
            "Dual-store: SQLite for exact/structured queries, Chroma for semantic "
            "similarity search (see the Company Memory tab) -- both wired into the "
            "real graph's human-in-the-loop finalize step, not just standalone."
        )


# ===========================================================================
# TAB 5 -- More Tools: PDF resume upload (multimodal ingestion) + calendar
# conflict check. Two more previously-built, previously-invisible pieces.
# Deliberately does NOT import mcp_server.py directly -- that module pulls
# in the `mcp` package, whose exact import path was already flagged as a
# risk; calendar_store.py underneath check_calendar_conflict has no such
# dependency, so this calls it directly instead.
# ===========================================================================
with tab_tools:
    st.header("Resume Upload -- PDF Document Understanding")
    st.caption(
        "Upload a PDF resume to see the project's multimodal ingestion path -- Gemini "
        "native document understanding, with a local OCR fallback if that fails. This "
        "only extracts text for display; it does NOT change which resume the Score a "
        "Job tab scores against. Uses a small amount of shared API quota."
    )

    uploaded_pdf = st.file_uploader("Upload a PDF resume", type=["pdf"])
    if uploaded_pdf is not None and st.button("Extract Text", type="primary"):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            with st.spinner("Extracting text (Gemini document understanding, OCR fallback if needed)..."):
                extraction = resume_document.parse_resume_document(tmp_path)

            if extraction["status"] == "ok":
                st.success(f"Extracted via: {extraction['source']}")
                st.text_area("Extracted text", value=extraction["extracted_text"], height=250)
            else:
                st.error(f"Extraction failed: {extraction.get('error')}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    st.divider()
    st.header("Calendar Conflict Check")
    st.caption(
        "Checks a candidate interview date against previously scheduled interviews. "
        "No API calls -- purely local logic. Booking data is separate from the "
        "Score a Job flow (that flow doesn't currently call this automatically)."
    )

    check_date = st.date_input("Interview date to check")
    check_label = st.text_input("What to schedule if the date is free (optional)")

    if st.button("Check Conflict"):
        date_str = check_date.isoformat()
        existing = calendar_store.get_events_on(date_str)

        if existing:
            st.warning(f"Conflict on {date_str}: {existing}")
            alternatives = calendar_store.suggest_alternatives(date_str, count=3)
            st.write("Suggested alternative dates:", alternatives)
        elif check_label:
            calendar_store.add_event(date_str, check_label)
            st.success(f"No conflict -- booked '{check_label}' on {date_str}.")
        else:
            st.success(f"No conflict on {date_str}.")
