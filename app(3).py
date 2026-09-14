import pickle
import numpy as np
import streamlit as st
import keras
from keras.utils import pad_sequences

# =========================================================
# CONFIGURATION — MUST MATCH THE TRAINING CODE
# =========================================================
MODEL_PATH = "bangla_real.keras"
TOKENIZER_PATH = "tokenizer_real.pickle"
MAX_LEN = 50

# Dataset label mapping:
# 0 = Negative
# 1 = Positive
# 2 = Neutral
CLASS_NAMES = [
    "নেগেটিভ (Negative)",
    "পজিটিভ (Positive)",
    "নিউট্রাল (Neutral)",
]

# Your training code used pad_sequences(sequences, maxlen=50)
# without specifying padding/truncating, so Keras defaults are:
# padding='pre', truncating='pre'
PADDING = "pre"
TRUNCATING = "pre"


# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================
@st.cache_resource(show_spinner="মডেল ও tokenizer লোড হচ্ছে...")
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# =========================================================
# PREDICTION
# =========================================================
def predict(text, model, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding=PADDING,
        truncating=TRUNCATING,
    )

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class]) * 100

    return predicted_class, confidence, probabilities, sequences[0]


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(
    page_title="বাংলা political Sentiment Analyzer ",
    page_icon="🇧🇩",
    layout="centered",
)

st.title("🇧🇩 বাংলা political Sentiment Analyzer")
st.write("রাজনৈতিক,অর্থনৈতিক এরকম কিছু লিখুন")

try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error("মডেল বা tokenizer লোড করা যাচ্ছে না।")
    st.code(str(e))
    st.info(
        "নিশ্চিত করো যে bangla_real.keras এবং tokenizer_real.pickle "
        "এই app(3).py-এর একই repository/folder-এ আছে।"
    )
    st.stop()

text = st.text_area(
    "বাংলা লেখা লিখুন👇",
    height=180,
    placeholder="দেশটা রসাতলে যাচ্ছে!",
)

if st.button("🔍 Sentiment Predict", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("দয়া করে আগে কিছু বাংলা লেখা লিখো।")
    else:
        pred_idx, confidence, probabilities, token_sequence = predict(
            text, model, tokenizer
        )

        st.subheader("📊 Prediction")
        st.success(
            f"**{CLASS_NAMES[pred_idx]}** — Confidence: **{confidence:.2f}%**"
        )

        st.write("### সব ক্লাসের সম্ভাবনা")
        for i, (name, probability) in enumerate(zip(CLASS_NAMES, probabilities)):
            st.write(f"**{name}**")
            st.progress(float(probability))
            st.caption(f"{float(probability) * 100:.2f}%")

        if len(token_sequence) == 0:
            st.warning(
                "Tokenizer এই লেখার কোনো শব্দ চিনতে পারেনি। "
                "এই অবস্থায় prediction কম নির্ভরযোগ্য হতে পারে।"
            )

        with st.expander("🔧 Technical details"):
            st.write("Tokenized sequence:", token_sequence)
            st.write("Padded input shape:", (1, MAX_LEN))
            st.write(
                "Raw probabilities:",
                [round(float(p), 6) for p in probabilities],
            )

st.divider()
st.caption("Model: Developed by Md. Nazmul Hasan Khan Mahmud. Bidirectional LSTM • Vocabulary: 20,000 • Sequence length: 50")
