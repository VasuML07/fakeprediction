#pandas is user for handling datasets
import pandas as pd
#numpy is used for numerical operartions
import numpy as np
#searborn is used for data visulaization advanced plots
import seaborn as sns
#matplotlib is used for basic plotting
import matplotlib.pyplot as plt
#this is used to split the dataset into train and test sets
from sklearn.model_selection import train_test_split
#used for converting the text input t numerical vectors using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
#for evaluating model performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#combines sparese matrices important dor tf-idf and extar features
from scipy.sparse import hstack
#used for saving out trained model and model weights
import pickle
print("Loading datasets...")
#loads the 3 datasets job scams fake news and real news
df_jobs = pd.read_csv("fake_job_postings.csv")
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")
#combines multiple colums into one text field and fillna replaces missing values with empty string
df_jobs["text"] = (
    df_jobs["title"].fillna('') + " " +
    df_jobs["location"].fillna('') + " " +
    df_jobs["description"].fillna('')
)
#removes rows where fraudelent is missing
df_jobs = df_jobs.dropna(subset=["fraudulent"])
#converts labels to integer 0 or 1
df_jobs["fraudulent"] = df_jobs["fraudulent"].astype(int)
#kees only relevant columsn
df_jobs = df_jobs[["text", "fraudulent"]]
#merges title+content into one columsn
df_fake["text"] = df_fake["title"] + " " + df_fake["text"]
df_real["text"] = df_real["title"] + " " + df_real["text"]
#labels fake news as 1 and real news as 0
df_fake["fraudulent"] = 1
df_real["fraudulent"] = 0
#combines both datasets for better results
df_news = pd.concat([df_fake, df_real], axis=0)
df_news = df_news[["text", "fraudulent"]]
#balamces datsets ensures that dataset size of news and jobs os equal
df_news = df_news.sample(len(df_jobs), random_state=42) 
#merges job+news datasets 
df = pd.concat([df_jobs, df_news], axis=0).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total samples: {len(df)}")
print("Creating features...")
#these are the keywords commonly found in scams according to datasets 
suspicious_words = [
    "earn money", "quick money", "no experience",
    "work from home", "easy job", "urgent hiring",
    "limited time", "apply now", "no interview",
    "instant hiring", "free registration", "investment required"
]
df["text_length"] = df["text"].apply(len)
#tells no of words
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
#this tells how many scam keywords appear
df["suspicious_score"] = df["text"].apply(
    lambda x: sum(word in x.lower() for word in suspicious_words)
)
df["caps_ratio"] = df["text"].apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
)
print("Vectorizing text...")
#converts etxt to numbers max 6000 features and uses unigrams and bigrams 
#unigrams means single and bigrams means pair of consecutive words
tfidf = TfidfVectorizer(
    max_features=6000,
    stop_words="english",
    ngram_range=(1, 2)
)
#learns vocabulary and transforms text
X_text = tfidf.fit_transform(df["text"])
#extarcts manual features
extra_features = df[[
    "text_length",
    "word_count",
    "suspicious_score",
    "caps_ratio"
]].values
#combines tf-idf+custom features
X = hstack([X_text, extra_features])
y = df["fraudulent"]
#train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print("Training model...")
#svm model gives more importance of fake class
model = LinearSVC(class_weight={0:1, 1:2}) 
model.fit(X_train, y_train)
#evaluation metrics
print("\n--- Evaluation ---")
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
#confusion matrix
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
#saves model weights as a pkl file
with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("Done!")
