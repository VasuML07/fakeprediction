#used for devoloping interactive web for ml focused applications
import streamlit as st
#used for saving and loading model
import pickle
#image module used for image specific functions
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

#sets configuration for age
#page_title is browser's tab tile
#layout is used to keep content in which manner
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)

# --------------------------------------------------
# Load Model & Vectorizer
# --------------------------------------------------
#this is used for heavy models,vectorizers and databases and loades once per session and doesn't rerun on each session
#this decorator is used for running function ones and storing input in cache
@st.cache_resource
def load_components():
    #loads serialized ml model
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)
    #loads our tf-idf vectorizer model
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

#ensures streamlit doesnt crash
try:
    model, vectorizer = load_components()
#throws and exception error on a crash
except FileNotFoundError:
    st.error("Model files not found. Run `train_model.py` first.")
    #stops the app from running
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}. Try retraining the model with the updated packages.")
    st.stop()

# --------------------------------------------------
# Sidebar – Model Info
# --------------------------------------------------

#sets title for the sidebar
st.sidebar.title("Model Information")
#writes the content in the sidebar
st.sidebar.write(
    """
    **Model:** NLP-based Fake Job Detector  
    **Technique:** TF-IDF + Naive Bayes  
    **Purpose:** Identify fraudulent job postings
    """
)

#checkbocks is a type of button only runs when user asks for it
if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        image = Image.open("confusion_matrix.png")
        #keeps that specific image with caption 
        st.sidebar.image(image, caption="Confusion Matrix")
    #handles error
    except FileNotFoundError:
        st.sidebar.warning("Confusion matrix not found. Train the model first.")

#writes some content in sidebar
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

#this draws a horizontal line b/w sections
st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

#used for user actions and steps
st.subheader("Enter Job Details")
#used for collecting inputs from user
job_title = st.text_input("Job Title")
job_location = st.text_input("Location (optional)")
job_description = st.text_area(
    "Job Description / Requirements",
    #height = 200 is for 200 chars per line to avoid one line screening
    height=200
)

# Combine input exactly like training
input_text = f"{job_title} {job_location} {job_description}"

# --------------------------------------------------
# Prediction
# --------------------------------------------------

#creates a button
if st.button("Analyze Job Posting", type="primary"):
    #if both inputs aren't entered we rasise a warning
    if not job_title and not job_description:
        st.warning("Please enter at least a job title or description.")
    else:
        #this fits out use input text into vectorizer to convert it to number
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

