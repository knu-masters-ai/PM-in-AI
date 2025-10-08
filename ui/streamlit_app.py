import base64
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="KidneyStoneAI (mock)",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    /* Трохи «медичної» стилізації */
    .stApp header { background: transparent; }
    .stDownloadButton > button, .stButton > button {
        border-radius: 10px;
        padding: 0.6rem 1rem;
        border: 1px solid #d9e6f2;
    }
    .stDownloadButton > button:hover, .stButton > button:hover {
        box-shadow: 0 2px 10px rgba(42,157,244,0.25);
        border-color: #2A9DF4;
    }
    .st-emotion-cache-16idsys p, p {
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 KidneyStoneAI (mock)")
st.caption("Upload a kidney scan image — mocked model will classify and highlight stones.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption="Original", use_column_width=True)
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    with st.spinner("Analyzing (mock)..."):
        response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
        data = response.json()
    st.success(f"Result: {data['label']} (confidence {data['confidence']:.2f})")
    st.write(data.get("message", ""))
    img_b64 = data["image_base64"]
    img_bytes = base64.b64decode(img_b64)
    st.image(img_bytes, caption="Annotated", use_column_width=True)
    st.download_button("Download annotated image", img_bytes, "annotated.png", "image/png")
else:
    st.info("Upload an image to start.")
