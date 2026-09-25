import pandas as pd
import joblib

df = pd.read_csv("dataset/cleaned_data.csv")

vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/fake_news_model.pkl")

fake_news = df[df["label"] == 0].sample(5, random_state=42)

texts = fake_news["content"]
features = vectorizer.transform(texts)
predictions = model.predict(features)

for i, (text, prediction) in enumerate(zip(texts, predictions), 1):
    actual = "FAKE" if fake_news.iloc[i - 1]["label"] == 0 else "REAL"
    predicted = "FAKE" if prediction == 0 else "REAL"

    print(f"\nExample {i}")
    print("Actual:", actual)
    print("Predicted:", predicted)
    print("Text:", text[:300])