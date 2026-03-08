# Reliable Multimodal Math Mentor AI

A production-ready AI application capable of solving JEE-style math problems using RAG, Multi-Agent Reasoning (LangGraph), Human-in-the-loop (HITL), and long-term memory.

## Architecture Diagram

```mermaid
graph TD
    A[Streamlit UI]
    A -->|Image| B(EasyOCR Pipeline)
    A -->|Audio| C(Whisper Pipeline)
    A -->|Text| D(Text Pipeline)
    
    B --> E{Confidence > Threshold?}
    C --> E
    D --> F[Text Input]
    
    E -->|No| G(HITL Correction)
    E -->|Yes| F
    G --> F
    
    F --> H[LangGraph Coordinator]
    
    H --> I[Parser Agent]
    I --> J[Intent Router Agent]
    J --> K[Solver Agent + RAG + SymPy]
    K --> L[Verifier Agent]
    
    L -->|Calculation Error or Edge Case| K
    L -->|Verification Confidence Low| G
    L -->|Correct| M[Explainer Agent]
    
    M --> N[SQLite Memory System]
    N --> A
```

## Setup Instructions

1. **Clone the repo**
2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your keys.
5. **Run the App**:
   ```bash
   streamlit run ui/app.py
   ```

## Deployment Steps
- **Streamlit Cloud**: Connect your GitHub repository to Streamlit Cloud, specify `ui/app.py` as the main entry point, and copy the `.env` contents to Streamlit secrets.
- **HuggingFace Spaces**: Create a Streamlit space, copy the repository, add the secrets in the Space settings. Add a `packages.txt` if system dependencies are needed (e.g., ffmpeg for Whisper).

## Overview
- **Agents**: Uses LangGraph to orchestrate Parser, Router, Solver, Verifier, and Explainer agents.
- **RAG Pipeline**: Retrieves geometric, algebraic, or calculus formulas from FAISS / Sentence-Transformers.
- **Memory**: Stores human feedback, input images, and previous results.
- **HITL (Human in the Loop)**: Any OCR failure, audio transcription ambiguity, or low verification confidence prompts human interaction in the Streamlit UI.
