"""
graph_viz.py

Live graph visualization, per spec section 3.8. Two distinct things:

  1. save_static_diagram() -- renders the graph's shape (nodes + edges) as
     a PNG via LangGraph's built-in Mermaid renderer. Useful for a README,
     but only shows the graph's overall structure, not what happened on
     any particular run.

  2. get_execution_path(thread_id) / print_execution_path(thread_id) --
     inspects the graph's checkpoint history for a specific thread_id and
     returns the actual ordered sequence of nodes that ran for that run.
     This is the real "watch the graph execute" artifact the spec asks
     for: "showing exactly which nodes/agents ran for a given input -- not
     just a static architecture diagram."

Both are read-only against the already-compiled `app` from
graph_builder.py -- this file doesn't modify the graph, how it runs, or
any existing file.
"""
from graph_builder import app


def save_static_diagram(path: str = "graph_diagram.png") -> str:
    """Renders the graph's static structure to a PNG using LangGraph's
    Mermaid renderer (which calls out to mermaid.ink to actually rasterize
    the PNG). Falls back to saving the raw Mermaid source as .mmd if that
    network call fails -- rendering failure shouldn't be able to crash a
    real run just because this got called from somewhere in a script.
    Returns the path actually written (PNG or .mmd fallback)."""
    graph = app.get_graph()
    try:
        png_bytes = graph.draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path
    except Exception as e:
        fallback_path = path.rsplit(".", 1)[0] + ".mmd"
        with open(fallback_path, "w") as f:
            f.write(graph.draw_mermaid())
        print(
            f"[WARN] PNG rendering failed ({e}); saved Mermaid source to "
            f"{fallback_path} instead. Paste its contents into "
            f"https://mermaid.live to view it."
        )
        return fallback_path


def get_execution_path(thread_id: str) -> list:
    """Returns the ordered list of node names that actually ran for a given
    thread_id, oldest-first -- reconstructed from the checkpointer's state
    history. get_state_history() returns snapshots newest-first, and each
    snapshot's metadata carries a `writes` dict keyed by the node name(s)
    that produced it (the initial pre-run snapshot has no writes, which is
    why `writes` is checked for truthiness rather than assumed present)."""
    config = {"configurable": {"thread_id": thread_id}}
    history = list(app.get_state_history(config))

    path = []
    for snapshot in reversed(history):
        writes = (snapshot.metadata or {}).get("writes")
        if writes:
            path.extend(writes.keys())
    return path


def print_execution_path(thread_id: str) -> None:
    """Human-readable version of get_execution_path -- prints the sequence
    as an arrow chain, or a clear message if the thread has no history."""
    path = get_execution_path(thread_id)
    if not path:
        print(f"No execution history found for thread_id={thread_id!r}.")
        return
    print(f"Execution path for thread_id={thread_id!r}:")
    print(" -> ".join(path))


if __name__ == "__main__":
    saved_path = save_static_diagram()
    print(f"Static graph diagram saved to {saved_path}")
