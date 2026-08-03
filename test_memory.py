"""
test_memory.py -- DEPRECATED.

This file originally tested memory.py's flat-JSON store directly. Since
Phase 8, nodes.py's memory_lookup_node/finalize_node read and write through
structured_memory.py (SQLite) + vector_memory.py (Chroma) instead --
memory.py is still in the repo but is no longer used by the running graph.

Running this file's old logic would write to memory.py's JSON store and
then read through nodes.memory_lookup_node (which now reads from
structured_memory.py instead) -- the two would silently disagree, making
it look like a bug ("I just saved a note but the lookup returns nothing")
when it's actually just testing two disconnected stores.

Use these instead:
    tests/test_memory_layer.py   -- structured_memory.py + vector_memory.py
                                     + migrate_memory.py, in isolation
    tests/test_memory_wiring.py  -- confirms nodes.py's finalize_node /
                                     memory_lookup_node actually use them

This file is kept only so `python test_memory.py` doesn't 404 for anyone
still running it out of habit -- it does nothing destructive, just points
here and exits.
"""
print(__doc__)
