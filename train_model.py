# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack
import pickle

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# =========================
# PREPROCESSING
# =========================
df["text"] = (
    df["title"].fillna('') + " " +
    df["location"].fillna('') + " " +
    df["description"].fillna('')
)

df = df.dropna(subset=["fraudulent"])
df["fraudulent"] = df["fraudulent"].astype(int)

# =========================
# FEATURE ENGINEERING
# =========================
print("Creating additional features...")

suspicious_words = [
    "earn money", "quick money", "no experience",
    "work from home", "easy job", "urgent hiring"
]

df["text_length"] = df["text"].apply(len)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))

df["suspicious_score"] = df["text"].apply(
    lambda x: sum(word in x.lower() for word in suspicious_words)
)

df["caps_ratio"] = df["text"].apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)

# =========================
# TF-IDF
# =========================
print("Vectorizing text...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text = tfidf.fit_transform(df["text"])

extra_features = df[[
    "text_length",
    "word_count",
    "suspicious_score",
    "caps_ratio"
]].values

X = hstack([X_text, extra_features])
y = df["fraudulent"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================
print("Training Linear SVM model...")
model = LinearSVC()
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

print("Confusion matrix saved.")

# =========================
# SAVE FILES
# =========================
print("Saving model and vectorizer...")

with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Done!")
