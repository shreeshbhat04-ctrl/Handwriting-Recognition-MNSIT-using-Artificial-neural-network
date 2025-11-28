import streamlit as st
from PIL import Image

from Trocr_engine import load_model_pipeline, run_trOCR


# ---------------------------------------------------------
#  Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwriting OCR",
    layout="wide",
)

# ---------------------------------------------------------
#  Global styling (no big background image)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: radial-gradient(circle at top, #1f2933 0, #020617 55%, #000000 100%);
    }

    /* Main content card */
    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        background: #020617ee;  /* solid dark, slight transparency */
        border-radius: 1.1rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #ffffff;
        font-weight: 800;
    }

    /* Force dark mode uploader */
div[data-testid="stFileUploader"] section {
    background-color: #0f172a !important;    /* dark navy */
    border: 1px dashed #ff6b6b !important;   /* your accent */
    border-radius: 0.8rem !important;
}

/* Fix text color (label + drag/drop text) */
div[data-testid="stFileUploader"] * {
    color: #f1f5f9 !important;   /* bright text */
    opacity: 1 !important;       /* force visibility */
}

/* Fix the upload icon */
div[data-testid="stFileUploader"] svg {
    fill: #f1f5f9 !important;
    opacity: 1 !important;
}

/* Fix "Browse files" button */
div[data-testid="stFileUploader"] button {
    color: #0f172a !important;
    background: #ffffff !important;
    border-radius: 0.6rem !important;
    font-weight: 600 !important;
}

    /* Buttons */
    .stButton>button {
        background: #f97373;
        color: #ffffff;
        border-radius: 0.7rem;
        border: none;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.15s ease-in-out;
    }
    .stButton>button:hover {
        background: #fb8c8c;
        transform: translateY(-1px);
    }

    /* Textarea dark theme */
    textarea {
        background-color: #020617 !important;
        color: #f9fafb !important;
        border-radius: 0.6rem !important;
    }

    /* Small caption text */
    .caption-text {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
#  Sidebar
# ---------------------------------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write(
    """
This app uses **Microsoft TrOCR** to convert handwritten
text images into editable text.

**Steps:**
1. Upload an image of handwriting  
2. Click **Extract Text**  
3. Copy or download the recognized text
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Clear, high-contrast handwriting works best.")

# ---------------------------------------------------------
#  Load model (cached)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model_pipeline("microsoft/trocr-large-handwritten")

with st.spinner("Loading AI model (only once)…"):
    processor, model = get_model()

st.success("Model ready! Upload an image to begin.")

# ---------------------------------------------------------
#  Main layout
# ---------------------------------------------------------
st.title("Handwriting to Text")
st.markdown("Turn handwritten notes into **searchable, editable text** using TrOCR.")

uploaded_file = st.file_uploader(
    "Upload a handwriting image",
    type=["jpg", "jpeg", "png"],
    help="Use a cropped image that mostly contains handwriting.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_text = st.columns([1.1, 1.3])

    # ---------- Left: image preview ----------
    with col_img:
        st.subheader("Preview")
        st.image(image, use_container_width=True)
        st.markdown(
            f"<p class='caption-text'>File: {uploaded_file.name} • "
            f"{image.width}×{image.height}px</p>",
            unsafe_allow_html=True,
        )

        resize = st.checkbox(
            "Resize long edge to 1024px (recommended for very large images)",
            value=True,
        )

        if resize:
            max_side = 1024
            w, h = image.size
            scale = min(max_side / max(w, h), 1.0)
            if scale < 1.0:
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size)

    # ---------- Right: OCR result ----------
    with col_text:
        st.subheader("Recognized Text")

        if st.button("Extract Text", type="primary"):
            with st.spinner("Analyzing handwriting…"):
                try:
                    result_text = run_trOCR(image, processor, model)
                    if not result_text.strip():
                        st.warning("No text detected. Try a clearer or closer image.")
                    else:
                        st.text_area("Output", result_text, height=260)
                        st.download_button(
                            "⬇️ Download as .txt",
                            data=result_text,
                            file_name="handwriting_ocr.txt",
                            mime="text/plain",
                        )
                except Exception as e:
                    st.error(f"An error occurred while running OCR: {e}")
        else:
            st.info("Click **Extract Text** to run OCR on the uploaded image.")
else:
    st.markdown(
        "<p class='caption-text'>No file uploaded yet. Use the uploader above to start.</p>",
        unsafe_allow_html=True,
    )
