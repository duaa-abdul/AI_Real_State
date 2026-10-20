
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"
GUIDE_FILE = BASE_DIR / "knowledge_base" / "property_guide.txt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_client = None
_model = None
_collection = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def get_collection():
    global _client, _collection

    if _collection is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)

        _client = chromadb.PersistentClient(path=str(DB_DIR))

        _collection = _client.get_or_create_collection(
            name="real_estate_knowledge"
        )

    return _collection


def split_text(text, chunk_size=180, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_knowledge_base():
    collection = get_collection()

    if not GUIDE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {GUIDE_FILE}"
        )

    text = GUIDE_FILE.read_text(encoding="utf-8")
    chunks = split_text(text)

    if not chunks:
        raise ValueError("Knowledge guide is empty.")

    ids = [f"property-guide-{i}" for i in range(len(chunks))]

    embeddings = get_embedding_model().encode(
        chunks, normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "property_guide.txt"} for _ in chunks],
    )

    return {"chunks_saved": len(chunks)}


def retrieve_context(question, top_k=3):
    collection = get_collection()
    count = collection.count()

    if count == 0:
        build_knowledge_base()
        count = collection.count()

    if count == 0:
        return []

    query_embedding = get_embedding_model().encode(
        [question], normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(documents, metadatas)
    ]
