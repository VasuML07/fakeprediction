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
print("Loading datasets...")

df_jobs = pd.read_csv("fake_job_postings.csv")

df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# =========================
# PREPROCESS JOB DATA
# =========================
df_jobs["text"] = (
    df_jobs["title"].fillna('') + " " +
    df_jobs["location"].fillna('') + " " +
    df_jobs["description"].fillna('')
)

df_jobs = df_jobs.dropna(subset=["fraudulent"])
df_jobs["fraudulent"] = df_jobs["fraudulent"].astype(int)
df_jobs = df_jobs[["text", "fraudulent"]]

# =========================
# PREPROCESS NEWS DATA
# =========================
df_fake["text"] = df_fake["title"] + " " + df_fake["text"]
df_real["text"] = df_real["title"] + " " + df_real["text"]

df_fake["fraudulent"] = 1
df_real["fraudulent"] = 0

df_news = pd.concat([df_fake, df_real], axis=0)
df_news = df_news[["text", "fraudulent"]]

# =========================
# BALANCE + COMBINE
# =========================
df_news = df_news.sample(len(df_jobs), random_state=42)  # avoid dominance

df = pd.concat([df_jobs, df_news], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total samples: {len(df)}")

# =========================
# FEATURE ENGINEERING
# =========================
print("Creating features...")

suspicious_words = [
    "earn money", "quick money", "no experience",
    "work from home", "easy job", "urgent hiring",
    "limited time", "apply now", "no interview",
    "instant hiring", "free registration", "investment required"
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
    max_features=6000,
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
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# MODEL
# =========================
print("Training model...")

model = LinearSVC(class_weight={0:1, 1:2})  # bias toward catching fake
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
print("\n--- Evaluation ---")

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# =========================
# SAVE
# =========================
with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Done!")
