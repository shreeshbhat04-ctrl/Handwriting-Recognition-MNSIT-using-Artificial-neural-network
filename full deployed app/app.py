import streamlit as st
from PIL import Image
from Trocr_engine import recognize_handwriting

st.set_page_config(page_title="AI Handwriting Reader", layout="centered")

st.title("Deep Learning Handwriting Recognition")
st.markdown("Using **Microsoft TrOCR** (Vision Transformer + BERT)")

uploaded_file = st.file_uploader("Upload handwritten text", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image
    st.image(image, caption='Your Upload', use_column_width=True)
    
    if st.button("Read Handwriting"):
        with st.spinner("The AI is reading... (This is heavier than Tesseract!)"):
            try:
                text = recognize_handwriting(image)
                
                st.success("Result:")
                st.markdown(f"### {text}")
                
            except Exception as e:
                st.error(f"Error: {e}")

st.info("Note: This runs on CPU by default. It will be slow (2-5 seconds). On a GPU, it is near-instant.")