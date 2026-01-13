#used for reading datasets
import pandas as pd
#used for mathematicl formulas
import numpy as np
#used for advanced visulization plots
import seaborn as sns
#used for some basic plottings
import matplotlib.pyplot as plt
#used for splitting the data into training and testing data
from sklearn.model_selection import train_test_split
#this used to convert out input text into integer values so our computer can understqand and process it
from sklearn.feature_extraction.text import TfidfVectorizer
#this multinomialNaive bayes algo preduicts based on proability of each frequency of words etc
from sklearn.naive_bayes import MultinomialNB
#used for model evaluation and insights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#used to save the model
import pickle

# 1. LOAD DATA
print("Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# 2. PREPROCESSING
#here we are selecting features for our input data and filling missing values in it
df["text"] = (
    df["title"].fillna('') + " " +
    df["location"].fillna('') + " " +
    df["description"].fillna('')
)

#here we are dropping fraudlent column from dataset because we need to predict it if we use it model memorizes
df = df.dropna(subset=["fraudulent"])
#converting fraudlent column data's type to int 
df["fraudulent"] = df["fraudulent"].astype(int)

# 3. VECTORIZATION (NLP STEP)
#used for converting text to numbers 
#max_features tells to only consider top 5000 features from the input helpful for memory
#stopwords removes useless words from vocabulary like the,is,of
#ngram_range = (1,2) means telling vectorizer to use unigrams for job and bi grams for 2 word phrases
print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)  # improves NLP performance
)

#dividing the datset into features and target
X = tfidf.fit_transform(df["text"])
y = df["fraudulent"]

# 4. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. MODEL TRAINING (NLP CLASSIFIER)
print("Training Naive Bayes NLP Model...")
model = MultinomialNB()
#fitting the model
model.fit(X_train, y_train)

# 6. MODEL EVALUATION
print("\n--- Model Evaluation Metrics ---")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#plotting a heatmap
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
print("Confusion Matrix saved as image.")

# 7. SAVE MODEL & VECTORIZER
print("\nSaving model and vectorizer...")
with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Done! Files saved.")

