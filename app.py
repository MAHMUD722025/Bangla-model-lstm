import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Bangla Sentiment Analysis using Bidirectional LSTM
# Developed by Md. Nazmul Hasan Khan Mahmud
# -------------------------------------------------

MODEL_PATH = "bangla_lstm.keras"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LEN = 50

st.set_page_config(
    page_title="বাংলা Sentiment Analysis",
    page_icon="🇧🇩",
    layout="centered"
)

st.title("🇧🇩 বাংলা Sentiment Analysis")
st.write("Bidirectional LSTM দিয়ে বাংলা লেখার sentiment বিশ্লেষণ করুন।")
st.caption("Developed by Md. Nazmu Hasan Khan Mahmud")

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"Model বা tokenizer load করা যাচ্ছে না: {e}")
    st.stop()

text = st.text_area(
    "বাংলা লেখা লিখুন:",
    placeholder="উদাহরণ: এই সিনেমাটি খুবই ভালো লেগেছে।",
    height=150
)

if st.button("🔍 Sentiment Predict", use_container_width=True):
    if not text.strip():
        st.warning("দয়া করে কিছু বাংলা লেখা লিখুন।")
    else:
        # IMPORTANT: inference preprocessing must match training.
        # Keras pad_sequences defaults are pre-padding and pre-truncation.
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="pre",
            truncating="pre"
        )

        prediction = float(model.predict(padded, verbose=0)[0][0])

        st.subheader("Prediction")
        st.progress(min(max(prediction, 0.0), 1.0))
        st.write(f"**Score:** {prediction:.4f}")

        if prediction >= 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

st.markdown("---")
st.caption("MD. Nazmul Hasan Khan Mahmud | Bangla Sentiment Analysis")
