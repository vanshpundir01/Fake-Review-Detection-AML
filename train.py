import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading dataset...")

df = pd.read_csv("new_data_test.csv")

df = df.dropna(subset=["text_", "label"])

df["text_"] = df["text_"].astype(str).apply(clean_text)

# CG = FAKE = 0, OR = REAL = 1
df["label"] = df["label"].astype(str).str.upper().str.strip()
df["label"] = df["label"].map({
    "CG": 0,
    "OR": 1
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

df = df[df["text_"].str.len() > 10]
df = df.drop_duplicates(subset=["text_", "label"])

print("\nLabel Count:")
print(df["label"].value_counts())

X = df["text_"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Extra fake samples only in training
extra_fake_reviews = [
    ("best best best product ever buy now", 0),
    ("amazing amazing amazing highly recommended", 0),
    ("perfect miracle product guaranteed results", 0),
    ("ultimate life changing product must buy", 0),
    ("buy now best product in the world", 0),
    ("excellent excellent excellent item", 0),
    ("guaranteed satisfaction limited offer", 0),
    ("must buy immediately unbelievable product", 0),
    ("best quality best price best service", 0),
    ("highly recommended amazing amazing product", 0),
    ("best best best product in the world must buy now", 0),
    ("amazing product amazing product amazing product", 0),
    ("perfect perfect perfect highly recommended", 0),
    ("unbelievable unbelievable best product buy now", 0),
    ("five star five star five star amazing product", 0),
]

extra_df = pd.DataFrame(extra_fake_reviews, columns=["text_", "label"])
extra_df["text_"] = extra_df["text_"].apply(clean_text)

X_train = pd.concat([X_train.reset_index(drop=True), extra_df["text_"]], ignore_index=True)
y_train = pd.concat([y_train.reset_index(drop=True), extra_df["label"]], ignore_index=True)

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

print("\nTraining model...")

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"\nTraining Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

train_df = pd.DataFrame({
    "text_": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text_": X_test,
    "label": y_test
})

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

joblib.dump(model, "model.pkl")

print("\nSaved:")
print("- model.pkl")
print("- train_data.csv")
print("- test_data.csv")