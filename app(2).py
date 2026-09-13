"""
Bangla Text Classifier — Streamlit App
---------------------------------------
এই অ্যাপটা তোমার trained model (bangla_real.keras) আর
tokenizer (tokenizer_real.pickle) ব্যবহার করে বাংলা টেক্সট ক্লাসিফাই করে।

Model architecture (ফাইল থেকে অটো-ডিটেক্ট করা হয়েছে):
    Embedding(vocab=20000, dim=80, input_length=50)
    -> Bidirectional(LSTM 64, return_sequences=True) -> Dropout(0.3)
    -> Bidirectional(LSTM 64)                          -> Dropout(0.3)
    -> Dense(3, activation="softmax")

চালানোর নিয়ম:
    1) pip install -r requirements.txt
    2) app.py, bangla_real.keras, tokenizer_real.pickle -- তিনটা ফাইল একই ফোল্ডারে রাখো
    3) streamlit run app.py
"""

import pickle

import numpy as np
import streamlit as st
import keras
from keras.utils import pad_sequences

# ----------------------------------------------------------------------------
# কনফিগারেশন — দরকার হলে এখানেই বদলাও
# ----------------------------------------------------------------------------
MODEL_PATH = "bangla_real.keras"
TOKENIZER_PATH = "tokenizer_real.pickle"
MAX_LEN = 50  # মডেলের input shape (None, 50) থেকে নেওয়া

# ⚠️ গুরুত্বপূর্ণ: মডেলের আউটপুট 3-ক্লাসের softmax, কিন্তু ফাইলের ভেতর ক্লাসের
# আসল নাম সংরক্ষিত থাকে না। Training-এর সময় label encode করার order অনুযায়ী
# নিচের লিস্টের ক্রম ঠিক করে নাও (index 0, 1, 2)। এখানে সবচেয়ে কমন
# sentiment-analysis convention (Negative, Neutral, Positive) ধরে নেওয়া হলো।
CLASS_NAMES = ["নেগেটিভ (Negative)", "নিউট্রাল (Neutral)", "পজিটিভ (Positive)"]

# Tokenizer.texts_to_sequences এর ফলাফল pad করার সময় কোন দিকে padding/truncation
# হবে সেটা training script-এ যা ব্যবহার হয়েছিল সেটার সাথে মিলিয়ে নাও।
PADDING = "post"
TRUNCATING = "post"


# ----------------------------------------------------------------------------
# মডেল ও টোকেনাইজার লোড (একবারই লোড হবে, cache করা আছে)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="মডেল লোড হচ্ছে...")
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)

    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except ModuleNotFoundError as e:
        st.error(
            "Tokenizer pickle লোড করতে সমস্যা হয়েছে (Keras version mismatch)। "
            f"আসল error: {e}\n\n"
            "সমাধান: `pip install --upgrade keras` করে আবার চেষ্টা করো, অথবা "
            "যে keras version দিয়ে tokenizer save হয়েছিল সেটাই ইনস্টল রাখো।"
        )
        st.stop()

    return model, tokenizer


def predict(model, tokenizer, text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING
    )
    probs = model.predict(padded, verbose=0)[0]
    return probs, seq[0]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Bangla Text Classifier", page_icon="📝", layout="centered")

st.title("📝 Bangla Text Classifier")
st.caption("BiLSTM মডেল দিয়ে বাংলা টেক্সট ক্লাসিফিকেশন — তোমার নিজের trained model থেকে")

model, tokenizer = load_artifacts()

text = st.text_area(
    "তোমার বাংলা টেক্সটটা এখানে লেখো 👇",
    height=150,
    placeholder="উদাহরণ: জিনিসপত্রের দাম অনেক বেড়ে গেছে, জীবন চালানো কঠিন হয়ে যাচ্ছে...",
)

col1, col2 = st.columns([1, 1])
predict_clicked = col1.button("প্রেডিক্ট করো 🚀", use_container_width=True, type="primary")
col2.button("ক্লিয়ার", use_container_width=True, on_click=lambda: st.session_state.clear())

if predict_clicked:
    if not text.strip():
        st.warning("দয়া করে আগে কিছু টেক্সট লেখো।")
    else:
        probs, seq = predict(model, tokenizer, text)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100

        st.subheader("ফলাফল")
        st.success(f"**{CLASS_NAMES[pred_idx]}**  —  {confidence:.1f}% কনফিডেন্স")

        st.write("প্রতিটা ক্লাসের সম্ভাবনা:")
        for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
            st.write(name)
            st.progress(float(p))
            st.caption(f"{float(p) * 100:.2f}%")

        if len(seq) == 0:
            st.info(
                "টোকেনাইজার এই বাক্যের কোনো শব্দই চিনতে পারেনি (vocabulary-তে নেই), "
                "তাই প্রেডিকশন নির্ভরযোগ্য নাও হতে পারে।"
            )

        with st.expander("🔍 ডিবাগ ইনফো"):
            st.write("Tokenized sequence:", seq)
            st.write("Padded shape:", (1, MAX_LEN))
            st.write("Raw probabilities:", [round(float(p), 4) for p in probs])

st.divider()
st.caption(
    "⚠️ নোট: CLASS_NAMES আর PADDING/TRUNCATING সেটিংস কোড-এর উপরের দিকে (কনফিগারেশন "
    "সেকশনে) আছে — তোমার training script-এ যেভাবে করেছিলে ঠিক সেভাবে মিলিয়ে নাও, "
    "নাহলে প্রেডিকশনের লেবেল/অ্যাকুরেসি ভুল আসতে পারে।"
)
