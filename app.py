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
# Load Model & Vectorizer
# --------------------------------------------------
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
    st.error("Model files not found. Run `train_model.py` first.")
    st.stop()
except Exception:
    st.error("Model files are corrupted. Retrain the model.")
    st.stop()

# --------------------------------------------------
# Sidebar – Model Info
# --------------------------------------------------
st.sidebar.title("Model Information")
st.sidebar.write(
    """
    **Model:** NLP-based Fake Job Detector  
    **Technique:** TF-IDF + Naive Bayes  
    **Purpose:** Identify fraudulent job postings
    """
)

if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        image = Image.open("confusion_matrix.png")
        st.sidebar.image(image, caption="Confusion Matrix", use_column_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Confusion matrix not found. Train the model first.")

st.sidebar.markdown(
    """
    **Metrics Explained**
    - **Precision:** How many predicted fakes are actually fake
    - **Recall:** How many real fakes were detected
    - **F1-Score:** Balance between Precision & Recall
    """
)

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🕵️ Fake Job Posting Detector")
st.markdown(
    "Detect whether a job posting is **Real** or **Fraudulent** using **NLP & Machine Learning**."
)

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("Enter Job Details")

job_title = st.text_input("Job Title")
job_location = st.text_input("Location (optional)")
job_description = st.text_area(
    "Job Description / Requirements",
    height=200
)

# Combine input exactly like training
input_text = f"{job_title} {job_location} {job_description}"

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Analyze Job Posting", type="primary"):
    if not job_title and not job_description:
        st.warning("Please enter at least a job title or description.")
    else:
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 FRAUDULENT JOB POSTING DETECTED")
            st.write(f"Confidence: **{probabilities[1] * 100:.2f}%**")
            st.info(
                "This posting contains linguistic patterns commonly seen in fake job ads. Proceed with caution."
            )
        else:
            st.success("✅ REAL JOB POSTING")
            st.write(f"Confidence: **{probabilities[0] * 100:.2f}%**")
            st.balloons()
