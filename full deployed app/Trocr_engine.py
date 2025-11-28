from PIL import Image, ImageOps
import re
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from my_timer import my_timer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/trocr-base-printed"   # better for letters+digits

def load_model_pipeline(model_name: str = MODEL_NAME):
    print(f"Loading model: {model_name} on {device}...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    return processor, model

def _preprocess(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((512, 128))        # simple, fixed size
    return img.convert("RGB")

@my_timer
def run_trOCR(image: Image.Image, processor, model) -> str:
    img = _preprocess(image)
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=32,
            num_beams=5,
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    text = re.sub(r"[^A-Za-z0-9]", "", text)   # keep only letters+digits
    return text
