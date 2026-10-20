
from app.embedding_service import (
    build_knowledge_base,
    retrieve_context,
)


def initialize_rag():
    return build_knowledge_base()


def get_relevant_context(question, top_k=3):
    return retrieve_context(question, top_k=top_k)
