import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# --------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------
df["text"] = (
    df["title"].fillna("") + " " +
    df["location"].fillna("") + " " +
    df["description"].fillna("")
)

df = df.dropna(subset=["fraudulent"])
df["fraudulent"] = df["fraudulent"].astype(int)

# --------------------------------------------------
# 3. VECTORIZATION (UPGRADED)
# --------------------------------------------------
print("Vectorizing text...")

tfidf = TfidfVectorizer(
    max_features=15000,
    stop_words="english",
    ngram_range=(1, 2),        # bigrams = big win
    min_df=3,
    sublinear_tf=True
)

X = tfidf.fit_transform(df["text"])
y = df["fraudulent"]

# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# 5. MODEL + HYPERPARAMETER TUNING
# --------------------------------------------------
print("Training Logistic Regression with tuning...")

param_grid = {
    "C": [0.01, 0.1, 1, 5, 10]
}

grid = GridSearchCV(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    ),
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print(f"Best C value: {grid.best_params_['C']}")

# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------
print("\n--- Model Evaluation Metrics ---")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------
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
plt.tight_layout()
plt.savefig("confusion_matrix.png")

print("Confusion Matrix saved as image.")

# --------------------------------------------------
# 8. SAVE MODEL & VECTORIZER
# --------------------------------------------------
print("\nSaving model and vectorizer...")

with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Done! Improved model saved.")
