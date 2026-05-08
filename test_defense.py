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
        "good": "g00d",
        "great": "gr8",
        "amazing": "amaz1ng",
        "best": "b3st",
        "nice": "nyc",
        "excellent": "excelent",
        "quality": "qu@lity",
        "service": "serv1ce",
        "product": "pr0duct",
        "delivery": "delivry",
        "packaging": "packagng",
        "food": "f00d",
        "friendly": "fr1endly",
        "quick": "qu1ck"
    }

    words = text.split()
    attacked_words = []

    for word in words:
        clean_word = word.strip(".,!?").lower()
        attacked_words.append(replacements.get(clean_word, word))

    attacked_words.insert(0, "honestly")
    attacked_words.append("totally recommended amazing buy now")

    return " ".join(attacked_words)


print("Loading test data and defense model...")

df = pd.read_csv("test_data.csv")
df = df.dropna(subset=["text_", "label"])

X_clean = df["text_"].astype(str).apply(clean_text)
X_attacked = df["text_"].astype(str).apply(strong_attack).apply(clean_text)
y_test = df["label"].astype(int)

model = joblib.load("defense_model.pkl")

print("\nEvaluating defense model on CLEAN data...")
pred_clean = model.predict(X_clean)

print("Accuracy (Clean):", accuracy_score(y_test, pred_clean))

print("\nEvaluating defense model on ATTACKED data...")
pred_attack = model.predict(X_attacked)

print("Accuracy (Attacked):", accuracy_score(y_test, pred_attack))

print("\nClassification Report (Attacked):\n")
print(classification_report(y_test, pred_attack))