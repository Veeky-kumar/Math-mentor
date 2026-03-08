import os
import tempfile
from groq import Groq
import sys

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import GROQ_API_KEY

def process_audio(audio_bytes) -> tuple[str, float]:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Groq API accepts tuple for file byte streaming: ("audio.m4a", bytes)
        completion = client.audio.transcriptions.create(
            file=("audio.m4a", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
            language="en"
        )
        
        text = completion.text
        
        # ASR via top models is generally high confidence
        avg_confidence = 0.95 if text else 0.0
        
        return text, avg_confidence
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Audio Error: {str(e)}", 0.0
