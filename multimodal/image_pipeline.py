import easyocr
import numpy as np
from PIL import Image

def process_image(image_bytes) -> tuple[str, float]:
    try:
        # Load reader (downloading models on first run)
        reader = easyocr.Reader(['en'], gpu=False)
        
        import io
        # Convert image bytes to np array
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        
        # Extract text
        results = reader.readtext(img_np)
        
        text_lines = []
        confidences = []
        
        for (bbox, text, prob) in results:
            text_lines.append(text)
            confidences.append(prob)
            
        full_text = " ".join(text_lines)
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.0
            
        return full_text, avg_confidence
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"OCR Error: {str(e)}", 0.0
