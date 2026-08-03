"""
MLflow tracing setup for the LangGraph agent (spec section 3.9).

What this buys, for one function call: every node execution and tool call
in the graph gets automatically captured into MLflow's trace viewer --
which node ran, how long it took, what it called, and (for LLM calls)
token usage and cost -- with zero manual instrumentation in nodes.py or
graph_builder.py beyond the one call this module makes.

View traces locally with:
    mlflow ui
then open http://localhost:5000 and pick the "job-search-copilot" experiment.

Deliberately NOT wired via a top-level import side effect in graph_builder.py
that silently fails hard if mlflow isn't installed -- enable_tracing() is
called explicitly and fails soft (prints a warning, doesn't crash the graph)
if mlflow is missing or autolog raises for any reason. Tracing is
observability, not a hard dependency the whole agent should die over.
"""
import os

_tracing_enabled = False


def enable_tracing(experiment_name: str = "job-search-copilot") -> bool:
    """Call once, before the graph runs. Idempotent -- safe to call more
    than once (e.g. once from graph_builder.py, once from a test script)
    without double-instrumenting. Returns True if tracing is active,
    False if it couldn't be enabled (missing mlflow, or any other error) --
    callers can check this if they want to know, but don't have to.
    """
    global _tracing_enabled
    if _tracing_enabled:
        return True

    try:
        import mlflow

        # Local, on-disk tracking store by default (SQLite, mlflow.db) --
        # no server, no account, no cost, unless MLFLOW_TRACKING_URI is
        # already set to something else in the environment. Newer MLflow
        # versions deprecate the plain filesystem store ("./mlruns") in
        # favor of a database backend, so this uses SQLite rather than a
        # bare folder path.
        if "MLFLOW_TRACKING_URI" not in os.environ:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment(experiment_name)
        mlflow.langchain.autolog()
        _tracing_enabled = True
        return True
    except Exception as e:
        print(f"[WARN] MLflow tracing could not be enabled (continuing without it): {e}")
        return False
