"""
react_from_scratch.py

Standalone proof-of-understanding: a hand-rolled ReAct loop
(Thought -> Action -> Observation -> ... -> Final Answer), no LangGraph, no
framework agent abstraction -- just raw LLM calls in a while loop, calling
one real tool (match_resume, reusing this project's actual
parse_node/score_node/RESUME_TEXT, not a mocked stand-in).

Deliberately separate from graph_builder.py's LangGraph pipeline: this
exists to demonstrate understanding of what LangGraph's ReAct-style agents
abstract over, not to replace the production graph.

Run directly:
    python react_from_scratch.py
"""
import re

from nodes import invoke_with_fallback, parse_node, score_node
from resume import RESUME_TEXT

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a job-fit assistant with exactly one tool available.

Tool: match_resume(job_description: str) -> str
  Scores a job description against the candidate's resume and returns a
  short summary of the fit score and matched/missing skills.

Respond in EXACTLY this format, one step at a time:
Thought: <your reasoning about what to do next>
Action: match_resume
Action Input: <the job description text to score>

Once you have the Observation back and are ready to answer, respond with:
Thought: <your reasoning>
Final Answer: <your final answer to the user>

Only ever take ONE action per turn. Wait for the Observation before continuing."""


def match_resume(job_description: str) -> str:
    """The one real tool this loop can call."""
    parsed = parse_node({"raw_input": job_description})
    scored = score_node(parsed, RESUME_TEXT)
    return (
        f"Fit score: {scored.get('fit_score')}. "
        f"Matched skills: {scored.get('matched_skills')}. "
        f"Missing skills: {scored.get('missing_skills')}."
    )


def parse_step(text: str) -> dict:
    """Extracts whichever of Action/Action Input or Final Answer is present
    in the LLM's raw response text."""
    action = re.search(r"Action:\s*(.+)", text)
    action_input = re.search(r"Action Input:\s*(.+)", text)
    final = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    return {
        "action": action.group(1).strip() if action else None,
        "action_input": action_input.group(1).strip() if action_input else None,
        "final_answer": final.group(1).strip() if final else None,
    }


def run_react_loop(user_query: str, verbose: bool = True) -> str:
    transcript = f"{SYSTEM_PROMPT}\n\nUser question: {user_query}\n"

    for i in range(MAX_ITERATIONS):
        response_text = invoke_with_fallback(transcript) or ""
        if verbose:
            print(f"\n--- Iteration {i + 1} ---\n{response_text}")

        step = parse_step(response_text)

        if step["final_answer"]:
            return step["final_answer"]

        if step["action"] == "match_resume" and step["action_input"]:
            observation = match_resume(step["action_input"])
            if verbose:
                print(f"Observation: {observation}")
            transcript += f"\n{response_text}\nObservation: {observation}\n"
        else:
            # Model didn't follow the format or asked for an unknown tool --
            # stop rather than loop forever on malformed output.
            return f"Stopped: could not parse a valid Action or Final Answer.\nLast response: {response_text}"

    return "Stopped: exceeded max iterations without a Final Answer."


if __name__ == "__main__":
    result = run_react_loop(
        "Score this job against my resume and tell me if I should apply: "
        "'AI Engineer role at Vertex Labs. Required: TensorFlow, Python, "
        "LangChain, RAG. Bachelor's in CS. 3+ years experience.'"
    )
    print(f"\n=== FINAL ANSWER ===\n{result}")
