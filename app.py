# =========================
# IMPORTS
# =========================
import streamlit as st
import pickle
import numpy as np
from PIL import Image
from scipy.sparse import hstack, csr_matrix

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fake Detection System",
    page_icon="🕵️",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_components():
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

try:
    model, vectorizer = load_components()
except FileNotFoundError:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Model Info")
st.sidebar.write("""
**Model:** Linear SVM  
**Technique:** TF-IDF + Feature Engineering  
**Data:** Job postings + News dataset  
""")

if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        image = Image.open("confusion_matrix.png")
        st.sidebar.image(image, caption="Confusion Matrix")
    except FileNotFoundError:
        st.sidebar.warning("Confusion matrix not found. Train the model first.")

st.sidebar.markdown("""
**Metrics Explanation**
- Precision → Correct fake predictions  
- Recall → Fake detection coverage  
- F1-score → Balance of both  
""")

# =========================
# MAIN UI
# =========================
st.title("🕵️ Fake Content Detector")
st.markdown(
    "Detect whether a **job posting or text content** is **Real or Fraudulent** using Machine Learning."
)

st.divider()

# =========================
# INPUT SECTION
# =========================
st.subheader("Enter Details")

job_title = st.text_input("Title (Job / News)")
job_location = st.text_input("Location (optional)")
job_description = st.text_area(
    "Description / Content",
    height=200
)

input_text = f"{job_title} {job_location} {job_description}"

# =========================
# FEATURE ENGINEERING
# =========================
def extract_features(text):
    suspicious_words = [
        "earn money", "quick money", "no experience",
        "work from home", "easy job", "urgent hiring"
    ]

    text_length = len(text)
    word_count = len(text.split())

    suspicious_score = sum(word in text.lower() for word in suspicious_words)

    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)

    return np.array([text_length, word_count, suspicious_score, caps_ratio]).reshape(1, -1)

# =========================
# PREDICTION
# =========================
if st.button("Analyze", type="primary"):

    if not job_title and not job_description:
        st.warning("Please enter at least a title or description.")
    else:
        try:
            # TF-IDF
            X_text = vectorizer.transform([input_text])

            # Extra features
            extra_features = extract_features(input_text)

            # Combine safely
            X_final = hstack([X_text, csr_matrix(extra_features)])

            # Prediction
            prediction = model.predict(X_final)[0]

            # Confidence (distance from boundary)
            decision = model.decision_function(X_final)[0]
            confidence = abs(decision)

            # =========================
            # OUTPUT
            # =========================
            st.divider()
            st.subheader("Result")

            if prediction == 1:
                st.error("🚨 FRAUDULENT / FAKE CONTENT DETECTED")
                st.info(
                    "This content shows patterns commonly associated with scams or misleading information."
                )
            else:
                st.success("✅ REAL / LEGIT CONTENT")
                st.balloons()

            st.write(f"Confidence Score: **{confidence:.2f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
