from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

class MathKnowledgeBase:
    def __init__(self, vector_db_path: str):
        self.vector_db_path = vector_db_path
        # Sentence-transformers (local)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._initialize_or_load()

    def _initialize_or_load(self):
        # Sample knowledge base including rules for algebra, calculus, probability, linear algebra, and common mistakes
        knowledge_docs = [
            Document(page_content="Algebra Identity: (a+b)^2 = a^2 + 2ab + b^2", metadata={"topic": "algebra"}),
            Document(page_content="Algebra Identity: a^2 - b^2 = (a-b)(a+b)", metadata={"topic": "algebra"}),
            Document(page_content="Calculus Derivative: d/dx(x^n) = n*x^(n-1)", metadata={"topic": "calculus"}),
            Document(page_content="Calculus Derivative: d/dx(sin(x)) = cos(x)", metadata={"topic": "calculus"}),
            Document(page_content="Probability: P(A|B) = P(A and B) / P(B)", metadata={"topic": "probability"}),
            Document(page_content="Probability: P(A or B) = P(A) + P(B) - P(A and B)", metadata={"topic": "probability"}),
            Document(page_content="Linear Algebra: Determinant of 2x2 matrix [[a,b],[c,d]] is ad - bc", metadata={"topic": "linear_algebra"}),
            Document(page_content="Common Mistake: (x+y)^2 is NOT x^2 + y^2. Remember the 2xy term.", metadata={"topic": "mistakes"}),
            Document(page_content="Calculus Product Rule: d/dx(u*v) = u * dv/dx + v * du/dx", metadata={"topic": "calculus"})
        ]
        
        # If the directory doesn't exist, create it and store docs
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
            self.vector_store = FAISS.from_documents(knowledge_docs, self.embeddings)
            self.vector_store.save_local(self.vector_db_path)
        else:
            try:
                self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                self.vector_store = FAISS.from_documents(knowledge_docs, self.embeddings)
                self.vector_store.save_local(self.vector_db_path)

    def retrieve(self, query: str, top_k: int = 3):
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Format the retrieved results with citation metadata
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "topic": doc.metadata.get("topic", "general"),
                "score": float(score)
            })
            
        return formatted_results
