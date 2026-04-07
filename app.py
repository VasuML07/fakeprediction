import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fake Detector", page_icon="🕵️", layout="centered")

# =========================
# CUSTOM CSS (UI MAGIC)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main {
    background-color: #0e1117;
}

h1 {
    font-size: 40px !important;
    font-weight: 700 !important;
}

.stTextInput input, .stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 12px;
    padding: 12px;
    border: none;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border: 1px solid #4CAF50;
}

.stButton button {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}

.stButton button:hover {
    background-color: #374151;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load():
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return model, vec

model, vectorizer = load()

# =========================
# HEADER
# =========================
st.markdown("# 🕵️ Fake Job Detector")

st.markdown("")

# =========================
# INPUT FIELDS
# =========================
title = st.text_input("Title")
location = st.text_input("Location")
desc = st.text_area("Description", height=200)

text = f"{title} {location} {desc}"

# =========================
# FEATURES
# =========================
def features(t):
    suspicious_words = [
        "earn money", "quick money", "no experience",
        "work from home", "easy job", "urgent hiring",
        "limited time", "apply now", "no interview",
        "instant hiring", "free registration", "investment required"
    ]

    return np.array([
        len(t),
        len(t.split()),
        sum(w in t.lower() for w in suspicious_words),
        sum(1 for c in t if c.isupper()) / (len(t)+1)
    ]).reshape(1, -1)

# =========================
# BUTTON
# =========================
if st.button("Analyze"):

    if not title and not desc:
        st.warning("Enter some content")
    else:
        X_text = vectorizer.transform([text])
        X_extra = features(text)

        X = hstack([X_text, csr_matrix(X_extra)])

        pred = model.predict(X)[0]
        score = abs(model.decision_function(X)[0])

        st.markdown("---")

        if pred == 1:
            st.error("🚨 FAKE CONTENT DETECTED")
        else:
            st.success("✅ REAL CONTENT")

        st.markdown(f"**Confidence Score:** `{score:.2f}`")
