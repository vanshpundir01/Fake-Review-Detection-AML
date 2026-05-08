import re
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading test data...")

df = pd.read_csv("test_data.csv")

df = df.dropna(subset=["text_", "label"])

X_test = df["text_"].astype(str).apply(clean_text)

y_test = df["label"].astype(int)

model = joblib.load("model.pkl")

print("\nTesting model...")

y_pred = model.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nSample Predictions:\n")

samples = [
    "The food was good and the service was quick",
    "Nice place to visit with friends and the staff was friendly",
    "Delivery was on time and packaging was good",
    "Amazing amazing amazing best product ever buy now",
    "Best best best product ever buy now",
    "Worst quality item waste of money"
]

for s in samples:
    pred = model.predict([clean_text(s)])[0]
    label = "FAKE" if pred == 0 else "REAL"
    print(f"{s} --> {label}")