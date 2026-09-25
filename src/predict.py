import joblib
import re

vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/fake_news_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

news = input("Enter news text: ")

news = clean_text(news)

news_tfidf = vectorizer.transform([news])

prediction = model.predict(news_tfidf)[0]
probability = model.predict_proba(news_tfidf)[0]

confidence = max(probability) * 100

if prediction == 1:
    result = "REAL"
else:
    result = "FAKE"

print("\nPrediction:", result)
print("Confidence:", round(confidence, 2), "%")