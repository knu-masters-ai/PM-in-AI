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
email_to = st.text_input("Email (optional):", placeholder="name@example.com")

# Ініціалізація session_state
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
    st.session_state.result = None           # JSON-відповідь з бекенду
    st.session_state.img_bytes = None        # анотований PNG (bytes)
    st.session_state.orig_bytes = None       # оригінал (bytes)
    st.session_state.emailed = None          # статус email
    st.session_state.email_error = None

if uploaded:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    st.image(uploaded, caption="Original", use_column_width=True)

    # Аналіз лише якщо новий файл (за хешем)
    if st.session_state.last_hash != file_hash:
        with st.spinner("Analyzing..."):
            try:
                # multipart: файл + (опційно) email як form-data
                files = {"file": (uploaded.name, file_bytes, uploaded.type or "application/octet-stream")}
                data = {"email": email_to} if email_to else None
                resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # Зберігаємо в session_state
        st.session_state.last_hash = file_hash
        st.session_state.result = result
        st.session_state.orig_bytes = file_bytes

        b64 = (result or {}).get("image_base64", "")
        try:
            st.session_state.img_bytes = base64.b64decode(b64) if b64 else None
        except Exception:
            st.session_state.img_bytes = None

        st.session_state.emailed = result.get("emailed", None)
        st.session_state.email_error = result.get("email_error")

    # --- Показ з кешу без повторного аналізу ---
    data = st.session_state.result or {}
    st.success(f"Result: {data.get('label', 'Unknown')}")
    st.write(data.get("message", ""))

    # Блок завантажень (оригінал + анотоване)
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.orig_bytes:
            st.download_button(
                "Download original image",
                data=st.session_state.orig_bytes,
                file_name=uploaded.name or "original.jpg",
                mime=uploaded.type or "image/jpeg",
                key="download_original",
            )
    with col2:
        if st.session_state.img_bytes:
            st.image(st.session_state.img_bytes, caption="Annotated", use_column_width=True)
            st.download_button(
                "Download annotated image",
                data=st.session_state.img_bytes,
                file_name="annotated.png",
                mime="image/png",
                key="download_annotated",
            )

    # Інформація про email-відправлення (якщо було)
    if st.session_state.emailed is True:
        st.success("Результат надіслано на email.")
    elif st.session_state.emailed is False:
        st.warning(f"Не вдалося надіслати email: {st.session_state.email_error or ''}")

    # Пояснення від AI (без confidence)
    exp = data.get("explanation") or {}
    if exp:
        st.subheader("Пояснення результату")
        if exp.get("summary_text"):
            st.write(exp["summary_text"])

        findings = exp.get("findings") or []
        if findings:
            st.markdown("**Знайдені ділянки:**")
            for i, f in enumerate(findings, 1):
                region = f.get("region", "(невідомо)")
                st.write(f"- **#{i}**: {region}")
                if f.get("evidence"):
                    st.caption(f"Пояснення: {f['evidence']}")

        steps = exp.get("next_steps") or []
        if steps:
            st.markdown("**Наступні кроки:**")
            for s in steps:
                st.write(f"- {s}")

        note = exp.get("clinical_note")
        if note:
            with st.expander("Clinical note"):
                st.write(note)
    else:
        st.info("Пояснювач тимчасово недоступний або не повернув структуру.")
else:
    st.info("Upload an image to start.")
