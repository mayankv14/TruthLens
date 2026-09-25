import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dataset/cleaned_data.csv")

df["content"] = df["content"].fillna("").astype(str)
df = df[df["content"].str.strip() != ""]
df = df.dropna(subset=["label"])

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/fake_news_model.pkl")

X_test_tfidf = vectorizer.transform(X_test)

predictions = model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nSample Predictions:")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

for i in range(10):
    actual = "REAL" if results.iloc[i]["Actual"] == 1 else "FAKE"
    predicted = "REAL" if results.iloc[i]["Predicted"] == 1 else "FAKE"

    print("\nActual:", actual)
    print("Predicted:", predicted)