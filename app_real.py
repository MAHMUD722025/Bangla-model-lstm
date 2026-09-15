import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_PATH = "bangla_real(2).keras"
TOKENIZER_PATH = "tokenizer_real(2).pickle"
MAX_LEN = 50

# The trained model has 3 output neurons (softmax).
# Keep this order consistent with the label encoding used during training.
CLASS_NAMES = [
    "নেগেটিভ (Negative)",
    "নিউট্রাল (Neutral)",
    "পজিটিভ (Positive)",
]

# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================
@st.cache_resource(show_spinner="মডেল ও tokenizer লোড হচ্ছে...")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# =========================================================
# PREDICTION
# =========================================================
def predict_sentiment(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])

    # IMPORTANT: must match the training preprocessing.
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]

    # Safety check in case the saved model has a different output size.
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model output has {len(probabilities)} classes, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} labels."
        )

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100

    return predicted_index, confidence, probabilities, sequence[0]


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="বাংলা Sentiment Analyzer",
    page_icon="🇧🇩",
    layout="centered",
)

st.title("🇧🇩 বাংলা Sentiment Analyzer")
st.caption("Bidirectional LSTM দিয়ে বাংলা টেক্সটের sentiment prediction")

try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error("মডেল বা tokenizer লোড করা যাচ্ছে না।")
    st.code(str(e))
    st.info(
        "নিশ্চিত করো যে bangla_real(2).keras এবং "
        "tokenizer_real(2).pickle একই repository/folder-এ আছে।"
    )
    st.stop()

text = st.text_area(
    "বাংলা লেখা লিখুন 👇",
    height=180,
    placeholder="উদাহরণ: এই পণ্যটির মান খুব ভালো, আমি অনেক খুশি।",
)

if st.button("🔍 Sentiment Predict", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("দয়া করে আগে কিছু বাংলা লেখা লিখুন।")
    else:
        try:
            pred_idx, confidence, probabilities, token_sequence = predict_sentiment(
                text, model, tokenizer
            )

            st.subheader("📊 Prediction")
            st.success(
                f"**{CLASS_NAMES[pred_idx]}** — Confidence: **{confidence:.2f}%**"
            )

            st.write("### সব ক্লাসের সম্ভাবনা")
            ranked = sorted(
                zip(CLASS_NAMES, probabilities),
                key=lambda x: float(x[1]),
                reverse=True,
            )

            for name, probability in ranked:
                st.write(f"**{name}**")
                st.progress(float(probability))
                st.caption(f"{float(probability) * 100:.2f}%")

            if len(token_sequence) == 0:
                st.warning(
                    "Tokenizer এই লেখার কোনো শব্দ চিনতে পারেনি। "
                    "এই prediction কম নির্ভরযোগ্য হতে পারে।"
                )

            with st.expander("🔧 Technical details"):
                st.write("Tokenized sequence:", token_sequence)
                st.write("Input shape:", (1, MAX_LEN))
                st.write(
                    "Raw probabilities:",
                    [round(float(p), 6) for p in probabilities],
                )

        except Exception as e:
            st.error("Prediction করার সময় সমস্যা হয়েছে।")
            st.code(str(e))

st.divider()
st.caption(
    "Model: Developed by Md. Nazmul Hasan Khan Mahmud • "
    "Bidirectional LSTM • Sequence length: 50"
)
