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
MAX_LEN = 80

# Binary sentiment classes:
# 0 = Negative
# 1 = Positive

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

    # The model expects a fixed sequence length of 80.
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    prediction = model.predict(padded, verbose=0)
    output = np.asarray(prediction).squeeze()

    # Case 1: Binary sigmoid output, e.g. [0.82]
    if np.ndim(output) == 0:
        positive_probability = float(output)
        positive_probability = max(0.0, min(1.0, positive_probability))
        negative_probability = 1.0 - positive_probability

    # Case 2: Two-neuron softmax output, [negative, positive]
    elif np.size(output) == 2:
        probabilities = np.asarray(output, dtype=float).reshape(-1)
        probabilities = probabilities / probabilities.sum()
        negative_probability = float(probabilities[0])
        positive_probability = float(probabilities[1])

    else:
        raise ValueError(
            f"এই binary app-এর জন্য model output shape সঠিক নয়: {np.shape(prediction)}"
        )

    predicted_class = 1 if positive_probability >= 0.5 else 0
    confidence = positive_probability if predicted_class == 1 else negative_probability

    return (
        predicted_class,
        confidence * 100,
        negative_probability,
        positive_probability,
        sequence[0],
    )


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="বাংলা বই রিভিউ Analyzer",
    page_icon="📚",
    layout="centered",
)

st.title("📚 বাংলা বই রিভিউ Analyzer")
st.caption("Bidirectional LSTM দিয়ে বাংলা বইয়ের রিভিউ Positive বা Negative হিসেবে বিশ্লেষণ করুন।")

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

review = st.text_area(
    "📖 বইয়ের রিভিউ লিখুন",
    height=200,
    placeholder="উদাহরণ: বইটির গল্প অসাধারণ। লেখকের ভাষা খুব সুন্দর এবং পড়তে অনেক ভালো লেগেছে।",
)

if st.button("🔍 রিভিউ বিশ্লেষণ করুন", use_container_width=True, type="primary"):
    if not review.strip():
        st.warning("দয়া করে আগে একটি বইয়ের রিভিউ লিখুন।")
    else:
        try:
            (
                predicted_class,
                confidence,
                negative_probability,
                positive_probability,
                token_sequence,
            ) = predict_sentiment(review, model, tokenizer)

            st.subheader("📊 ফলাফল")

            if predicted_class == 1:
                st.success(f"### 😊 Positive (1)\nConfidence: **{confidence:.2f}%**")
            else:
                st.error(f"### 😞 Negative (0)\nConfidence: **{confidence:.2f}%**")

            st.write("### ক্লাসের সম্ভাবনা")

            st.write("**Positive (1)**")
            st.progress(float(positive_probability))
            st.caption(f"{positive_probability * 100:.2f}%")

            st.write("**Negative (0)**")
            st.progress(float(negative_probability))
            st.caption(f"{negative_probability * 100:.2f}%")

            if len(token_sequence) == 0:
                st.warning(
                    "Tokenizer এই রিভিউয়ের কোনো পরিচিত শব্দ খুঁজে পায়নি। "
                    "তাই এই prediction কম নির্ভরযোগ্য হতে পারে।"
                )

            with st.expander("🔧 Technical details"):
                st.write("Tokenized sequence:", token_sequence)
                st.write("Input shape:", (1, MAX_LEN))
                st.write("Class mapping:", "0 = Negative, 1 = Positive")

        except Exception as e:
            st.error("রিভিউ বিশ্লেষণ করার সময় সমস্যা হয়েছে।")
            st.code(str(e))

st.divider()
st.caption(
    "Model: Developed by Md. Nazmul Hasan Khan Mahmud • "
    "Bidirectional LSTM • Binary Sentiment Classification • Sequence length: 80"
)
