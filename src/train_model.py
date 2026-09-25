import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv("dataset/cleaned_data.csv")

df["content"] = df["content"].fillna("").astype(str)

df = df[df["content"].str.strip() != ""]

df = df.dropna(subset=["label"])


# Features and labels
X = df["content"]
y = df["label"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english"
)


# Transform text
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(
    max_iter=2000,
    C=2.0
)

model.fit(X_train_tfidf, y_train)


# Predictions
predictions = model.predict(X_test_tfidf)


# Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


# Save model
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(model, "model/fake_news_model.pkl")


# Results
print("\n==============================")
print("MODEL TRAINING RESULTS")
print("==============================")

print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nModel training completed successfully!")