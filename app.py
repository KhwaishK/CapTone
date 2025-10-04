import streamlit as st
from pipeline import process_image, rephrase_caption
import yaml

# Page config
st.set_page_config(page_title="AI Captioning & Rephrasing Tool", layout="wide")

# Load config
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DEFAULT_IMAGE = params["ui"]["test_image"]
AVAILABLE_TONES = params["llm"]["tones"]

# Custom CSS styling (optional)
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    h1 { text-align: center; color: #4CAF50; font-size: 2.2em; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 12px; padding: 0.6em 1.2em; font-size: 16px; font-weight: bold; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.03); }
    .css-1d391kg { background-color: #1a1a2e !important; }
    </style>
""", unsafe_allow_html=True)

st.title("AI Captioning & Rephrasing Tool")

# Sidebar
st.sidebar.header("Settings")
selected_tone = st.sidebar.selectbox("Choose a tone", AVAILABLE_TONES)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    caption, hashtags = process_image(uploaded_file)
    image_to_show = uploaded_file
else:
    st.info("Using default test image")
    caption, hashtags = process_image(DEFAULT_IMAGE)
    image_to_show = DEFAULT_IMAGE

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


