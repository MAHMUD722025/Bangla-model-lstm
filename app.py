import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Bangla Sentiment Analysis using Bidirectional LSTM
# Developed by Shaykh Molla Lakshmipuri
# -------------------------------------------------

MODEL_PATH = "bangla_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

st.set_page_config(
    page_title="বাংলা Sentiment Analysis",
    page_icon="🇧🇩",
    layout="centered"
)

st.title("🇧🇩 বাংলা Sentiment Analysis")
st.write("Bidirectional LSTM দিয়ে বাংলা লেখার sentiment বিশ্লেষণ করুন।")
st.caption("Developed by Shaykh Molla Lakshmipuri")

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error("Model বা tokenizer load করা যাচ্ছে না। Repository-তে model এবং tokenizer file আছে কি না দেখুন।")
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
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

        prediction = float(model.predict(padded, verbose=0)[0][0])

        st.subheader("Prediction")
        st.progress(min(max(prediction, 0.0), 1.0))
        st.write(f"**Score:** {prediction:.4f}")

        if prediction >= 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

st.markdown("---")
st.caption("© Shaykh Molla Lakshmipuri | Bangla Sentiment Analysis")
