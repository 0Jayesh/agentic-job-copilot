"""
Chroma-backed semantic memory retrieval, sitting alongside
structured_memory.py (SQLite).

Split of responsibilities:
  - structured_memory.py answers "what exactly happened with Company X"
    (exact-match lookup, structured fields, queryable by score/status)
  - vector_memory.py answers "what past situations are similar to this one"
    (embedding similarity search across all notes, not just one company)

Reuses the same SentenceTransformer("all-MiniLM-L6-v2") model nodes.py
already loads for semantic_match, via `from nodes import embedder`, instead
of loading a second copy of the model into memory. Everything here is
local and free: Chroma runs as an on-disk persistent store
(chromadb.PersistentClient), no hosted service, no API key.

This file does NOT modify nodes.py, memory.py, or anything in the existing
graph -- same "new parallel layer" approach as structured_memory.py.
"""
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from nodes import embedder  # reuse the already-loaded SentenceTransformer

PERSIST_DIR = "chroma_memory"
COLLECTION_NAME = "company_memory"


class _SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Adapts nodes.py's SentenceTransformer instance to Chroma's
    EmbeddingFunction interface. Newer chromadb versions require more than
    just __call__: name()/get_config()/build_from_config() are used when
    Chroma persists/validates the embedding function alongside a
    collection, and are called even on a plain get_or_create_collection."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return embedder.encode(list(input)).tolist()

    @staticmethod
    def name() -> str:
        return "job_copilot_sentence_transformer_minilm"

    def get_config(self) -> dict:
        # No configurable parameters -- the model name is fixed in nodes.py.
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "_SentenceTransformerEmbeddingFunction":
        return _SentenceTransformerEmbeddingFunction()


_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_SentenceTransformerEmbeddingFunction(),
        )
    return _collection


def add_memory(
    company: str,
    note: str,
    fit_score: Optional[float] = None,
    status: Optional[str] = None,
    sql_row_id: Optional[int] = None,
) -> str:
    """Embeds and stores one note. `sql_row_id`, if given, links this Chroma
    entry back to its row in structured_memory.company_notes so the two
    stores can be cross-referenced by ID rather than drifting independently.
    Returns the Chroma document ID."""
    collection = _get_collection()
    doc_id = f"note-{sql_row_id}" if sql_row_id is not None else f"note-{datetime.now(timezone.utc).timestamp()}"

    metadata = {"company": company, "created_at": datetime.now(timezone.utc).isoformat()}
    if fit_score is not None:
        metadata["fit_score"] = fit_score
    if status is not None:
        metadata["status"] = status

    collection.upsert(
        ids=[doc_id],
        documents=[note],
        metadatas=[metadata],
    )
    return doc_id


def query_similar(query_text: str, n_results: int = 5) -> List[dict]:
    """Semantic search across ALL companies' notes -- 'find past situations
    like this one', not restricted to one company."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    n_results = min(n_results, collection.count())

    results = collection.query(query_texts=[query_text], n_results=n_results)

    out = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({"id": doc_id, "note": doc, "metadata": meta, "distance": dist})
    return out


def query_by_company(company: str, n_results: int = 10) -> List[dict]:
    """Semantic search scoped to one company's notes only, via a metadata
    filter rather than a full-corpus scan."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(where={"company": company})
    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    out = [{"id": i, "note": d, "metadata": m} for i, d, m in zip(ids, docs, metas)]
    return out[:n_results]
