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


def strong_attack(text):
    text = str(text).lower()

    replacements = {
        "good": "bad",
        "great": "terrible",
        "amazing": "worst",
        "best": "worst",
        "nice": "bad",
        "excellent": "poor",
        "perfect": "awful",
        "love": "hate",
        "liked": "hated",
        "friendly": "rude",
        "quick": "slow",
        "clean": "dirty",
        "fresh": "stale",
        "delicious": "tasteless",
        "recommend": "avoid",
        "awesome": "awful"
    }

    words = text.split()
    attacked_words = []

    for word in words:
        clean_word = word.strip(".,!?").lower()
        attacked_words.append(replacements.get(clean_word, word))

    attacked_words.insert(0, "fake sponsored review")
    attacked_words.append("do not trust this review")

    return " ".join(attacked_words)


print("Loading test data...")

df = pd.read_csv("test_data.csv")
df = df.dropna(subset=["text_", "label"])

X_test = df["text_"].astype(str)
y_test = df["label"].astype(int)

model = joblib.load("model.pkl")

print("\nEvaluating model on attacked data...")

X_attacked = X_test.apply(strong_attack).apply(clean_text)

y_pred = model.predict(X_attacked)

print("Accuracy on Adversarial Data:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nSample Attacked Predictions:\n")

samples = [
    "The food was good and the service was quick",
    "Nice place to visit with friends and the staff was friendly",
    "Delivery was on time and packaging was good",
    "Amazing amazing amazing best product ever buy now",
    "Best best best product ever buy now",
    "Worst quality item waste of money"
]

for s in samples:
    attacked = clean_text(strong_attack(s))
    pred = model.predict([attacked])[0]

    label = "FAKE" if pred == 0 else "REAL"

    print("Original:", s)
    print("Attacked:", attacked)
    print("Prediction:", label)
    print("-" * 50)