from rag.knowledge_base import MathKnowledgeBase
from config.settings import VECTOR_DB_DIR

def build_retriever():
    return MathKnowledgeBase(VECTOR_DB_DIR)

def retrieve_context(query: str):
    retriever = build_retriever()
    return retriever.retrieve(query)
