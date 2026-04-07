import streamlit as st
import pickle
import numpy as np
from PIL import Image
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Detector", page_icon="🕵️")

@st.cache_resource
def load():
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return model, vec

model, vectorizer = load()

st.title("🕵️ Fake Content Detector")

title = st.text_input("Title")
location = st.text_input("Location")
desc = st.text_area("Description", height=200)

text = f"{title} {location} {desc}"

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

if st.button("Analyze"):
    if not title and not desc:
        st.warning("Enter input")
    else:
        X_text = vectorizer.transform([text])
        X_extra = features(text)

        X = hstack([X_text, csr_matrix(X_extra)])

        pred = model.predict(X)[0]
        score = abs(model.decision_function(X)[0])

        if pred == 1:
            st.error("🚨 FAKE")
        else:
            st.success("✅ REAL")

        st.write(f"Confidence: {score:.2f}")
