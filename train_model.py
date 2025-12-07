import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. LOAD DATA
print("Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# 2. PREPROCESSING
# fillna replaces null values with empty strings
df["text"] = df["title"].fillna('') + " " + df["location"].fillna('') + " " + df["description"].fillna('')
df = df.dropna(subset=["fraudulent"])
df["fraudulent"] = df["fraudulent"].astype(int)

# 3. VECTORIZATION
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df["text"])
y = df["fraudulent"]

# 4. TRAIN/TEST SPLIT
# Fixed typo: 'startify' -> 'stratify'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. MODEL TRAINING
print("Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. MODEL EVALUATION METRICS (What you asked for)
print("\n--- Model Evaluation Metrics ---")
y_pred = model.predict(X_test)

# Calculate specific metrics
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Detailed Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png") # Save graph to display in Streamlit later
print("Confusion Matrix saved as image.")

# 7. SAVE ARTIFACTS
print("\nSaving model and vectorizer...")
with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Done! Files saved.")