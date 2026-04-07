import streamlit as st
import pickle
from PIL import Image
from scipy.sparse import hstack

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Fake Job Detector",
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
except:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Model Info")
st.sidebar.write("""
**Model:** Linear SVM  
**Technique:** TF-IDF + Feature Engineering  
""")

if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        image = Image.open("confusion_matrix.png")
        st.sidebar.image(image)
    except:
        st.sidebar.warning("Train model first")

# =========================
# MAIN UI
# =========================
st.title("🕵️ Fake Job Detector")
st.write("Detect fraudulent job postings using ML")

st.divider()

job_title = st.text_input("Job Title")
job_location = st.text_input("Location")
job_description = st.text_area("Job Description", height=200)

input_text = f"{job_title} {job_location} {job_description}"

# =========================
# FEATURE ENGINEERING (SAME AS TRAINING)
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

    return [text_length, word_count, suspicious_score, caps_ratio]

# =========================
# PREDICTION
# =========================
if st.button("Analyze", type="primary"):

    if not job_title and not job_description:
        st.warning("Enter job details")
    else:
        X_text = vectorizer.transform([input_text])
        extra = extract_features(input_text)

        X_final = hstack([X_text, [extra]])

        prediction = model.predict(X_final)[0]
        decision = model.decision_function(X_final)[0]

        confidence = abs(decision)

        st.divider()
        st.subheader("Result")

        if prediction == 1:
            st.error("🚨 FAKE JOB DETECTED")
        else:
            st.success("✅ REAL JOB")

        st.write(f"Confidence Score: {confidence:.2f}")
