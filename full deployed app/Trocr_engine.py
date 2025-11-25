from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the pre-trained model specifically for handwriting
# We use 'fast' to cache it so it doesn't reload on every click
@torch.no_grad()
def load_model():
    print("Loading TrOCR Model... this may take a minute.")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

# Initialize once
processor, model = load_model()

def recognize_handwriting(image):
    """
    Takes a PIL Image, processes it via Vision Transformer, 
    and decodes it into text.
    """
    # Convert image to RGB (just in case it's grayscale/RGBA)
    image = image.convert("RGB")

    # 1. Preprocess (Resize/Normalize - handled automatically by the processor)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 2. Generate Text (The model "looks" and predicts tokens)
    generated_ids = model.generate(pixel_values)
    
    # 3. Decode tokens back to string
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text