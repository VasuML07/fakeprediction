# 🕵️‍♂️ Fake Job Prediction System  
### NLP-Based Fraudulent Job Detection

An NLP-powered machine learning system designed to detect fraudulent job postings using text classification techniques.

This project leverages TF-IDF vectorization and a trained ML model to distinguish between legitimate job opportunities and scams, helping job seekers make safer decisions.

---

## 🔗 Live Demo

🔴 https://fakeprediction-a8wpvpp3uifhwxeduehaev.streamlit.app/

---

## 🚀 Core Functionality

- Analyze job description text
- Convert text into numerical features using TF-IDF
- Classify posting as:
  - ✅ Legitimate  
  - 🚫 Fraudulent  
- Real-time prediction via web interface  

---

## 🧠 Machine Learning Pipeline

**Text Input →  
Preprocessing →  
TF-IDF Vectorization →  
Trained Classification Model →  
Prediction Output**

---

## 🛠 Tech Stack

### 💻 Core Language
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

---

### 🤖 Machine Learning
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Text%20Vectorization-blue?style=flat-square)

---

### 🌐 Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 📂 Project Structure

fakeprediction/
│
├── app.py # Web interface (Streamlit)
├── fake_job_model.pkl # Trained classification model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── train_model.py # Model training script
└── requirements.txt # Dependencies


---

## ⚙️ Run Locally

### 📥 Clone Repository

```bash
git clone https://github.com/VasuML07/fakeprediction.git
cd fakeprediction
🧪 Create Virtual Environment
python -m venv venv
Activate:

Windows

venv\Scripts\activate
macOS / Linux

source venv/bin/activate
📦 Install Dependencies
pip install -r requirements.txt
▶ Run Application
streamlit run app.py
📊 Model Details
Text representation using TF-IDF

Supervised classification model

Trained on labeled job posting dataset

Designed to detect linguistic patterns associated with scams

🔮 Future Improvements
Use transformer-based embeddings (BERT)

Add explainability (feature importance visualization)

Improve dataset size and class balance

Deploy via Docker

Add user authentication & logging

