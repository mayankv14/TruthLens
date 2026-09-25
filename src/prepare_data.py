import pandas as pd
import re

df = pd.read_csv("dataset/WELFake_Dataset.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna(subset=["title", "text"])
df = df.drop_duplicates()

df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = df["content"].apply(clean_text)

df = df.dropna(subset=["content"])
df = df[df["content"].str.strip() != ""]

df = df[["content", "label"]]

print("\nAfter cleaning:")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

df.to_csv("dataset/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")