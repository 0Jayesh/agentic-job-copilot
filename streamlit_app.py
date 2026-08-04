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
import uuid

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from graph_builder import app as graph_app
import structured_memory
import graph_viz

st.set_page_config(page_title="Job Search Copilot", page_icon="🧭", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

tab_score, tab_memory, tab_graph = st.tabs(["Score a Job", "Company Memory", "Graph Diagram"])


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

            approve_col, reject_col = st.columns(2)
            if approve_col.button("Approve and send", type="primary"):
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

    company_query = st.text_input("Company name")
    if company_query:
        notes = structured_memory.get_company_memory_detailed(company_query)
        if notes:
            for row in notes:
                st.write(f"**{row.get('created_at', '')[:10]}** -- {row.get('note')}")
        else:
            st.write("No history found for this company yet.")

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
