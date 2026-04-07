#used to build the web app ui
import streamlit as st
#used to load saved ml model and vectorizer
import pickle
import numpy as np
#used to combine matxices and converst extat features into readable fomrat
from scipy.sparse import hstack, csr_matrix
#used to open and load images 
from PIL import Image
#sets browser tab title,icon and layout
st.set_page_config(page_title="Fake Detector", page_icon="🕵️", layout="centered")
#injects custom css into app 
st.markdown("""
<style>
body { background-color: #0e1117; }
.main { background-color: #0e1117; }
h1 { font-size: 40px !important; font-weight: 700 !important; }
.stTextInput input, .stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 12px;
    padding: 12px;
    border: none;
}
.stButton button {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}
.stButton button:hover {
    background-color: #374151;
}
</style>
""", unsafe_allow_html=True)
#cache function prevents reloading model evry time user interacts
@st.cache_resource
#function to load vectorizer and model
def load():
    with open("fake_job_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return model, vec
    #calls function and stores results
model, vectorizer = load()
#main heading
st.markdown("# 🕵️ Fake Job Detector")
#title
st.sidebar.title("📊 Model Analytics")
#creates toggle bar for seeing confuasion matxix
if st.sidebar.checkbox("Show Confusion Matrix"):
    try:
        #display saved image
        img = Image.open("confusion_matrix.png")
        st.sidebar.image(img, caption="Confusion Matrix")
    except:
        st.sidebar.warning("Run training first")
st.sidebar.markdown("""
**Model:** Linear SVM  
**Features:** TF-IDF + Custom signals  
**Signals Used:**
- Text length  
- Word count  
- Suspicious phrases  
- Capital ratio  
""")
#inpiuts
title = st.text_input("Title")
location = st.text_input("Location")
desc = st.text_area("Description", height=200)
#combines all inputs
text = f"{title} {location} {desc}"
suspicious_words = [
    "earn money", "quick money", "no experience",
    "work from home", "easy job", "urgent hiring",
    "limited time", "apply now", "no interview",
    "instant hiring", "free registration", "investment required"
]
def extract_features(t):
    #counts total char
    text_length = len(t)
    #counts total words
    word_count = len(t.split())
    #counts how many scam words found
    suspicious_score = sum(w in t.lower() for w in suspicious_words)
    caps_ratio = sum(1 for c in t if c.isupper()) / (len(t)+1)
    #returns feature array 
    return np.array([text_length, word_count, suspicious_score, caps_ratio]).reshape(1, -1), {
        "Text Length": text_length,
        "Word Count": word_count,
        "Suspicious Words": suspicious_score,
        "Caps Ratio": round(caps_ratio, 3)
    }
    #highlights suspicious words
def highlight_text(text):
    highlighted = text
    for word in suspicious_words:
        if word in text.lower():
            highlighted = highlighted.replace(word, f"🔴{word.upper()}")
    return highlighted
    #runs when user clicks button
if st.button("Analyze"):
    #prevenst empty input
    if not title and not desc:
        st.warning("Enter some content")
    else:
        #cobverts text to tf-idf vector
        X_text = vectorizer.transform([text])
        #Extract custom features
        X_extra, feature_dict = extract_features(text)
        #Combine text + numeric features
        X = hstack([X_text, csr_matrix(X_extra)])
        #Predict class: 1 → fake 0 → real
        pred = model.predict(X)[0]
        #Confidence score (distance from decision boundary)
        score = abs(model.decision_function(X)[0])
        st.markdown("---")
        if pred == 1:
            st.error("🚨 FAKE JOB DETECTED")
        else:
            st.success("✅ REAL JOB")
        st.markdown(f"**Confidence Score:** `{score:.2f}`")
        st.subheader("🔍 Feature Analysis")
        for k, v in feature_dict.items():
            st.write(f"**{k}:** {v}")
        st.subheader("🧠 Suspicious Pattern Highlight")
        st.markdown(f"""
        <div style="background-color:#262730;padding:10px;border-radius:10px">
        {highlight_text(text)}
        </div>
        """, unsafe_allow_html=True)
        st.subheader("⚡ Insight")
        if feature_dict["Suspicious Words"] > 2:
            st.warning("High number of scam-like phrases detected.")
        if feature_dict["Caps Ratio"] > 0.2:
            st.warning("Too many capital letters → possible spam signal.")
        if feature_dict["Word Count"] < 20:
            st.info("Very short description → less reliable job post.")
