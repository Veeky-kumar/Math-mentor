import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "conversations.db")

# Confidence thresholds
OCR_CONFIDENCE_THRESHOLD = 0.8
VERIFIER_CONFIDENCE_THRESHOLD = 0.8
