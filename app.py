import streamlit as st
import pickle
import numpy as np
from PIL import Image

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="JobShield | Fake Job Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 2. Load Model & Vectorizer (Cached)
# --------------------------------------------------
@st.cache_resource
def load_components():
    try:
        with open("fake_job_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_components()

# --------------------------------------------------
# 3. Sidebar - Stats & Info
# --------------------------------------------------
with st.sidebar:
    st.title("📊 Project Insights")
    st.info("This system uses Natural Language Processing (NLP) to analyze linguistic patterns in job postings.")
    
    if st.checkbox("Show Performance Metrics"):
        try:
            image = Image.open("confusion_matrix.png")
            st.image(image, caption="Model Confusion Matrix")
        except:
            st.warning("Confusion matrix image not found.")

    st.markdown("---")
    st.markdown("### How it works")
    st.write("1. **TF-IDF**: Converts text to weighted numbers.")
    st.write("2. **Naive Bayes**: Calculates probability of fraud based on word frequency.")

# --------------------------------------------------
# 4. Main UI Layout
# --------------------------------------------------
st.title("🕵️ Fake Job Posting Detector")
st.write("Enter the details of a job posting below to check if it matches fraudulent patterns.")

# Split the layout into two columns
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("prediction_form"):
        st.subheader("📋 Job Details")
        title = st.text_input("Job Title", placeholder="e.g. Data Entry Clerk")
        location = st.text_input("Location", placeholder="e.g. London, UK")
        description = st.text_area(
            "Job Description & Requirements", 
            placeholder="Paste the full job description here...",
            height=250
        )
        
        submit_button = st.form_submit_button("Analyze Posting")

# --------------------------------------------------
# 5. Prediction Logic
# --------------------------------------------------
with col2:
    st.subheader("🔍 Analysis Result")
    
    if submit_button:
        if not title or not description:
            st.error("Please fill in at least the Title and Description.")
        elif model is None or vectorizer is None:
            st.error("Model files missing! Please run the training script first.")
        else:
            # Prepare data (match training format)
            input_data = f"{title} {location} {description}"
            
            # Transform and Predict
            vec_input = vectorizer.transform([input_data])
            prediction = model.predict(vec_input)[0]
            probs = model.predict_proba(vec_input)[0]
            
            # Display Results
            if prediction == 1:
                st.error("### 🚨 HIGH RISK")
                st.write(f"This posting shows strong indicators of being **Fraudulent**.")
                
                # Confidence Progress Bar
                conf = probs[1]
                st.write(f"Confidence Level: **{conf*100:.1f}%**")
                st.progress(conf)
                
                st.warning("⚠️ **Advice:** Do not share personal bank details or pay any 'application fees'.")
            else:
                st.success("### ✅ LOW RISK")
                st.write("This posting appears to be **Legitimate** based on our analysis.")
                
                # Confidence Progress Bar
                conf = probs[0]
                st.write(f"Confidence Level: **{conf*100:.1f}%**")
                st.progress(conf)
                
                st.balloons()

    else:
        st.info("Enter job details and click 'Analyze Posting' to see the result.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and uses machine learning probabilities. Always do your own due diligence.")
