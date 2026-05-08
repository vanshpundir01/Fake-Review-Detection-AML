import re
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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


print("Loading training data...")

df = pd.read_csv("train_data.csv")
df = df.dropna(subset=["text_", "label"])

df["label"] = df["label"].astype(int)

clean_df = df.copy()
clean_df["text_"] = clean_df["text_"].astype(str).apply(clean_text)

attack_df = df.copy()
attack_df["text_"] = attack_df["text_"].astype(str).apply(strong_attack).apply(clean_text)

final_df = pd.concat([clean_df, attack_df], ignore_index=True)

X_train = final_df["text_"]
y_train = final_df["label"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        stop_words="english",
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=0.7,
        class_weight="balanced"
    ))
])

print("\nTraining defense model...")

model.fit(X_train, y_train)

pred = model.predict(X_train)

print(f"\nDefense Training Accuracy: {accuracy_score(y_train, pred):.4f}")

print("\nDefense Training Report:\n")
print(classification_report(y_train, pred))

joblib.dump(model, "defense_model.pkl")

print("\nDefense training complete.")
print("Saved: defense_model.pkl")