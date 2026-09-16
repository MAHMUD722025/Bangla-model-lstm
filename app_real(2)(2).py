"""
Bangla Sentiment Analysis - Streamlit App
-------------------------------------------
Loads a trained Keras (Bidirectional LSTM) model and its matching
Tokenizer to classify Bangla text as Positive or Negative sentiment.

Model summary (read from the .keras file):
    Input        -> (None, 80)                     # 80 tokens per review
    Embedding    -> input_dim=20000, output_dim=80
    Bidirectional LSTM(64, return_sequences=True)
    Dropout(0.3)
    Bidirectional LSTM(32)
    Dropout(0.3)
    Dense(1, activation="sigmoid")                  # binary classification

Files expected in the same folder as this script:
    bangla_real_2__2_.keras
    tokenizer_real_2__2_.pickle
"""

import pickle
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "bangla_real(2)(2).keras"
TOKENIZER_PATH = "tokenizer_real(2)(2).pickle"
MAX_LEN = 80  # must match the model's fixed input length

st.set_page_config(
    page_title="Bangla Sentiment Analyzer",
    page_icon="🇧🇩📝📚",
    layout="centered",
)

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


def clean_text(text: str) -> str:
    """Light cleanup before tokenizing (tokenizer itself already lowercases
    and strips the punctuation it was configured with, but this removes
    stray digits/URLs that can otherwise pollute short inputs)."""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def predict_sentiment(text: str, tokenizer, model):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")

    known_tokens = sum(1 for t in seq[0])
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if label == "Positive" else 1 - score
    return label, confidence, score, known_tokens


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🇧🇩📝📚Bangla bool review Sentiment Analyzer")
st.write(
    "Type or paste a Bangla review below and the model will predict whether "
    "the sentiment is **Positive** or **Negative**."
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        f"""
- **Architecture:** Embedding → Bidirectional LSTM(64) → Dropout → Bidirectional LSTM(32) → Dropout → Dense(1, sigmoid)
- **Vocabulary size:** up to {20000:,} tokens
- **Sequence length:** fixed at **{MAX_LEN}** tokens (longer text is truncated, shorter text is padded)
- This is a binary classifier — the output score close to **1** means Positive, close to **0** means Negative.
        """
    )

text_input = st.text_area(
    "Enter Bangla text",
    height=150,
    placeholder="এখানে আপনার বাংলা রিভিউ লিখুন...",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            model = load_model()
            tokenizer = load_tokenizer()
        except Exception as e:
            st.error(f"Could not load model/tokenizer: {e}")
        else:
            label, confidence, raw_score, known_tokens = predict_sentiment(
                text_input, tokenizer, model
            )

            if label == "Positive":
                st.success(f"**Sentiment: {label}** 😊")
            else:
                st.error(f"**Sentiment: {label}** 😞")

            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(
                f"Confidence: {confidence * 100:.2f}%  |  Raw sigmoid score: {raw_score:.4f}"
            )

            if known_tokens == 0:
                st.info(
                    "Note: none of the words in this text were found in the "
                    "model's vocabulary, so this prediction may not be reliable."
                )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "Developed by MD.Nazmul Hasan Khan Mahmud "
    "</div>",
    unsafe_allow_html=True,
)
