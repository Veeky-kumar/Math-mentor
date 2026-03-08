import streamlit as st
import uuid
import sys
import os

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multimodal.image_pipeline import process_image
from multimodal.audio_pipeline import process_audio
from agents.workflow import workflow
from memory.persistent_memory import PersistentMemory
from config.settings import OCR_CONFIDENCE_THRESHOLD, VERIFIER_CONFIDENCE_THRESHOLD

memory = PersistentMemory()

st.set_page_config(page_title="Math Mentor AI", layout="wide")

st.title("🎓 Reliable Multimodal Math Mentor AI")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

input_mode = st.radio("Select Input Mode", ("Text", "Image", "Audio"), horizontal=True)

initial_text = ""
confidence = 1.0

if input_mode == "Text":
    initial_text = st.text_area("Enter your Math problem:", height=100)

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image containing Math (jpg/png)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Extracting text via OCR..."):
            initial_text, confidence = process_image(uploaded_file.read())
        
        st.write(f"**OCR Confidence:** {confidence:.2f}")
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            st.warning("OCR confidence is low! Please review and fix the text below (HITL Checkpoint).")
        
        # HITL Text Box
        initial_text = st.text_area("Extracted Text (Edit if needed):", initial_text, height=100)

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload Audio Question (wav/mp3)", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file)
        with st.spinner("Transcribing..."):
            initial_text, confidence = process_audio(audio_file.read())
            
        st.write(f"**ASR Confidence:** {confidence:.2f}")
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            st.warning("ASR confidence is low! Please review and fix the text below (HITL Checkpoint).")
            
        initial_text = st.text_area("Transcribed Text (Edit if needed):", initial_text, height=100)

if st.button("Solve Problem"):
    if initial_text.strip() == "":
        st.error("Please provide a valid math problem.")
    else:
        # Check overall confidence trigger for HITL
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            st.warning("Low confidence input confirmed by user. Proceeding to Agent Workflow...")

        with st.spinner("Agents are reasoning..."):
            initial_state = {
                "original_text": initial_text,
                "parsed_problem": {},
                "topic": "",
                "retrieved_context": [],
                "solver_output": "",
                "is_verified": False,
                "verification_feedback": "",
                "verification_attempts": 0,
                "final_explanation": "",
                "hitl_required": False
            }
            
            # Execute LangGraph Workflow
            events = workflow.stream(initial_state)
            
            st.write("### Agent Execution Trace")
            for event in events:
                for node_name, node_state in event.items():
                    st.write(f"> **Executed Agent:** {node_name.capitalize()}")
                    
                    if node_name == "parser":
                        st.json(node_state.get("parsed_problem", {}))
                        
                    elif node_name == "router":
                        st.success(f"Topic Classified to: {node_state.get('topic')}")
                        
                    elif node_name == "retriever":
                        with st.expander("Retrieved RAG Context", expanded=False):
                            st.write(node_state.get("retrieved_context", []))
                            
                    elif node_name == "solver":
                        with st.expander("Solver Work", expanded=False):
                            st.write(node_state.get("solver_output", ""))
                            
                    elif node_name == "verifier":
                        is_verified = node_state.get("is_verified", False)
                        msg = "Pass" if is_verified else f"Fail: {node_state.get('verification_feedback')}"
                        st.info(f"Verification: {msg}")

        # Final state extraction
        # Because we only get pieces in stream, it's better to run invoke or reconstruct from stream. 
        # For simplicity, let's just run invoke right after or handle the state globally.
        final_state = workflow.invoke(initial_state)
        
        st.markdown("---")
        if final_state.get("hitl_required", False):
            st.error("⚠️ HITL TRiGGER: Workflow halted due to ambiguity or verification failure!")
            st.write("Verification Feedback: ", final_state.get("verification_feedback", "Ambiguous Problem input."))
        else:
            st.subheader("🎓   Final Explanation")
            st.write(final_state.get("final_explanation", "No explanation provided."))

            # Save to Memory
            memory.save_interaction(st.session_state.session_id, {
                "original_input": initial_text,
                "parsed_problem": final_state.get("parsed_problem"),
                "retrieved_context": final_state.get("retrieved_context"),
                "generated_solution": final_state.get("final_explanation"),
                "verification_result": final_state.get("verification_feedback")
            })

            # Feedback mechanism
            st.write("### Rate this Solution")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Correct"):
                    memory.update_feedback(st.session_state.session_id, "Correct")
                    st.success("Thanks for the feedback! Saved to memory.")
            with col2:
                if st.button("❌ Incorrect"):
                    memory.update_feedback(st.session_state.session_id, "Incorrect")
                    st.error("Feedback logged. Model will try to learn from this.")
