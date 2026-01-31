# ============================================================
# Fake Job Posting Detector
# A clean, stable UI wrapped around an NLP ML model
# ============================================================

import streamlit as st
import pickle
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)


# --------------------------------------------------
# Light UI Styling (kept minimal on purpose)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0f172a, #020617);
    color: #e5e7eb;
}

/* Main button */
.stButton>button {
    height: 3rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 10px;
}

/* Text areas */
textarea {
    line-height: 1.6 !important;
    font-size: 1rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Model & Vectorizer (cached, runs once)
# --------------------------------------------------
@st.cache_resource
def load_components():
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Guardrails so the app fails gracefully
try:
    model, vectorizer = load_components()
except FileNotFoundError:
    st.error("Model files not found. Run `train_model.py` first.")
    st.stop()
except Exception as e:
    st.error(
        f"Error loading model: {e}. "
        "Try retraining the model with updated packages."
    )
    st.stop()


# --------------------------------------------------
# Sidebar – Model Information (static, clean)
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Model Information")

    st.markdown(
        """
        **Model Type:** NLP Classifier  
        **Technique:** TF-IDF + Naive Bayes  
        **Goal:** Detect fraudulent job postings
        """
    )

    if st.checkbox("Show Confusion Matrix"):
        try:
            image = Image.open("confusion_matrix.png")
            st.image(image, caption="Confusion Matrix")
        except FileNotFoundError:
            st.warning("Confusion matrix not found. Train the model first.")

    st.markdown("---")
    st.markdown(
        """
        **Metrics Explained**
        - **Precision:** How accurate fake predictions are  
        - **Recall:** How many fake jobs were caught  
        - **F1-Score:** Balance of precision & recall
        """
    )


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown("## 🕵️ Fake Job Posting Detector")
st.markdown(
    "Analyze a job posting using **Natural Language Processing** "
    "to check whether it is **Real** or **Fraudulent**."
)

st.divider()


# --------------------------------------------------
# Input Section (structured, calm)
# --------------------------------------------------
st.subheader("📝 Enter Job Details")

job_title = st.text_input(
    "Job Title",
    placeholder="e.g., Data Analyst"
)

job_location = st.text_input(
    "Location (optional)",
    placeholder="e.g., Remote / New York"
)

job_description = st.text_area(
    "Job Description / Requirements",
    height=200,
    placeholder="Paste the full job description here..."
)

# Combine inputs exactly as during training
input_text = f"{job_title} {job_location} {job_description}"


# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
analyze_btn = st.button(
    "🔍 Analyze Job Posting",
    type="primary",
    use_container_width=True
)

if analyze_btn:
    if not job_title and not job_description:
        st.warning("Please enter at least a job title or job description.")
    else:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("🚨 FRAUDULENT JOB POSTING DETECTED")
            st.write(f"Confidence: **{probabilities[1] * 100:.2f}%**")
            st.info(
                "This posting shows linguistic patterns commonly found in fake job ads. "
                "Proceed with caution."
            )
        else:
            st.success("✅ REAL JOB POSTING")
            st.write(f"Confidence: **{probabilities[0] * 100:.2f}%**")
            st.balloons()
