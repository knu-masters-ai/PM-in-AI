import base64, os, requests, streamlit as st, hashlib
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="KidneyStoneAI",
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

st.title("🩺 KidneyStoneAI")
st.caption("Upload a kidney scan image — model will classify and highlight stones.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # ініціалізація state
    if "last_hash" not in st.session_state:
        st.session_state.last_hash = None
        st.session_state.result = None  # тут зберігатимемо JSON відповіді
        st.session_state.img_bytes = None

    # показ оригіналу
    st.image(uploaded, caption="Original", use_column_width=True)

    # Виконуємо аналіз лише якщо файл новий/інший
    if st.session_state.last_hash != file_hash:
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(f"{API_URL}/predict",
                                     files={"file": (
                                     uploaded.name, file_bytes, uploaded.type or "application/octet-stream")},
                                     timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        st.session_state.last_hash = file_hash
        st.session_state.result = data
        # дістаємо анотоване зображення у bytes
        b64 = (data or {}).get("image_base64", "")
        try:
            st.session_state.img_bytes = base64.b64decode(b64) if b64 else None
        except Exception:
            st.session_state.img_bytes = None

    # --- Відображаємо з кешу (без повторного аналізу) ---
    data = st.session_state.result or {}
    st.success(f"Result: {data.get('label', 'Unknown')}")
    st.write(data.get("message", ""))

    if st.session_state.img_bytes:
        st.image(st.session_state.img_bytes, caption="Annotated", use_column_width=True)
        st.download_button(
            "Download annotated image",
            data=st.session_state.img_bytes,
            file_name="annotated.png",
            mime="image/png",
            key="download_annotated",
        )

    exp = data.get("explanation") or {}
    if exp:
        st.subheader("Пояснення результату")
        if exp.get("summary_text"):
            st.write(exp["summary_text"])
        for i, f in enumerate(exp.get("findings") or [], 1):
            st.write(f"- **#{i}**: {f.get('region', '(невідомо)')}")
            if f.get("evidence"):
                st.caption(f"Пояснення: {f['evidence']}")
        steps = exp.get("next_steps") or []
        if steps:
            st.markdown("**Наступні кроки:**")
            for s in steps: st.write(f"- {s}")
        note = exp.get("clinical_note")
        if note:
            with st.expander("Clinical note"):
                st.write(note)
    else:
        st.info("Пояснювач тимчасово недоступний або не повернув структуру.")
else:
    st.info("Upload an image to start.")
