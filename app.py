import os

# Redirect caches to temporary folders (prevents storage overflow)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub"
os.makedirs("/tmp/hf_cache", exist_ok=True)
os.makedirs("/tmp/hf_home", exist_ok=True)
os.makedirs("/tmp/hf_hub", exist_ok=True)

# Prevent Streamlit permission issues
os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"
os.environ["STREAMLIT_CACHE_DIR"] = "/tmp/.streamlit_cache"
os.environ["STREAMLIT_DISABLE_USAGE_STATS"] = "true"
os.makedirs("/tmp/.streamlit", exist_ok=True)
os.makedirs("/tmp/.streamlit_cache", exist_ok=True)


import streamlit as st
from pipeline import process_image, rephrase_caption
import yaml

# -------------------------------
# Load configuration
# -------------------------------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DEFAULT_IMAGE = params["ui"]["test_image"]
AVAILABLE_TONES = params["llm"]["tones"]

# Page setup
st.set_page_config(page_title="AI Captioning & Rephrasing Tool", layout="wide")
st.title("📸 AI Captioning & Rephrasing Tool")

# -------------------------------
# Sidebar settings
# -------------------------------
st.sidebar.header("⚙️ Settings")
selected_tone = st.sidebar.selectbox("Choose a tone:", AVAILABLE_TONES)

# -------------------------------
# File upload section
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

# -------------------------------
# Session state initialization
# -------------------------------
if "caption" not in st.session_state:
    st.session_state.caption = None
if "hashtags" not in st.session_state:
    st.session_state.hashtags = []
if "image_path" not in st.session_state:
    st.session_state.image_path = DEFAULT_IMAGE

# -------------------------------
# Process image (with caching)
# -------------------------------
@st.cache_data(show_spinner=False)
def get_caption_and_tags(image):
    return process_image(image)

# If user uploads image
if uploaded_file is not None:
    st.session_state.image_path = uploaded_file
    st.session_state.caption, st.session_state.hashtags = get_caption_and_tags(uploaded_file)

# If no image uploaded, use default
elif st.session_state.caption is None:
    st.info("Using default test image")
    st.session_state.caption, st.session_state.hashtags = get_caption_and_tags(DEFAULT_IMAGE)

# -------------------------------
# Layout display
# -------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image(st.session_state.image_path, use_container_width=True)

with col2:
    with st.expander("🖋️ Generated Caption", expanded=True):
        st.write(st.session_state.caption)

    with st.expander("🏷️ Hashtags", expanded=True):
        st.write(", ".join(st.session_state.hashtags))

    with st.expander("🎭 Rephrased Caption", expanded=True):
        if st.button("✨ Rephrase"):
            tone_caption = rephrase_caption(st.session_state.caption, selected_tone)
            st.write(f"**{selected_tone} tone:** {tone_caption}")

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    # <-- updated parameter here
    st.image(image_to_show, caption="Input Image", use_container_width=True)

with col2:
    with st.expander("Generated Caption", expanded=True):
        st.write(caption)

    with st.expander("Hashtags", expanded=True):
        st.write(", ".join(hashtags))

    with st.expander("Rephrased Captions"):
        if st.button("Rephrase"):
            tone_caption = rephrase_caption(caption, selected_tone)
            st.success(f"**{selected_tone} tone:** {tone_caption}")


