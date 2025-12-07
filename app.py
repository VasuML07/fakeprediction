import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️", layout="centered")

# --- LOAD SAVED MODEL & VECTORIZER ---
@st.cache_resource
def load_components():
    # We open files in "rb" (Read Binary) mode to load them
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_components()
except FileNotFoundError:
    st.error("Model files not found! Please run 'train_model.py' first.")
    st.stop()
except EOFError:
    st.error("Model files are empty/corrupted! Please delete .pkl files and run 'train_model.py' again.")
    st.stop()

# --- SIDEBAR: METRICS ---
st.sidebar.title("Model Performance")
st.sidebar.write("This model uses **Logistic Regression**.")
if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        # We display the image saved during training
        image = Image.open('confusion_matrix.png')
        st.sidebar.image(image, caption='Confusion Matrix', use_column_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Run train_model.py to generate the matrix image.")

st.sidebar.markdown("""
**Metrics Explained:**
* **Precision:** How many selected items are relevant?
* **Recall:** How many relevant items are selected?
* **F1-Score:** Balance between Precision and Recall.
""")

# --- MAIN APP UI ---
st.title("🕵️ Fake Job Posting Detector")
st.markdown("### Detect if a job advertisement is Real or Fraudulent using AI.")

st.write("---")

# Input Section
st.subheader("Job Details")
job_title = st.text_input("Job Title")
job_desc = st.text_area("Job Description / Requirements", height=200)
job_loc = st.text_input("Location (Optional)")

# Combine inputs exactly like we did in training
input_text = f"{job_title} {job_loc} {job_desc}"

# Prediction Logic
if st.button("Analyze Job Posting", type="primary"):
    if not job_desc and not job_title:
        st.warning("Please enter at least a Job Title or Description.")
    else:
        # 1. Transform input using the loaded vectorizer
        transformed_input = vectorizer.transform([input_text])
        
        # 2. Predict
        prediction = model.predict(transformed_input)[0]
        probabilities = model.predict_proba(transformed_input)[0] # Get % confidence
        
        # 3. Display Results
        st.write("---")
        st.subheader("Prediction Result:")
        
        # 1 is Fake, 0 is Real (based on your dataset)
        if prediction == 1:
            st.error("🚨 FRAUDULENT JOB POSTING DETECTED")
            st.write(f"Confidence: **{probabilities[1]*100:.2f}%**")
            st.info("Be careful! This job post has patterns similar to known fake jobs.")
        else:
            st.success("✅ REAL JOB POSTING")
            st.write(f"Confidence: **{probabilities[0]*100:.2f}%**")
            st.balloons()