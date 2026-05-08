import re
import random
import joblib
import streamlit as st
import numpy as np


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_typo(text):
    text = str(text)
    if len(text) < 5:
        return text
    index = random.randint(0, len(text) - 1)
    return text[:index] + text[index + 1:]


def inject_words(text):
    words = str(text).split()
    if not words:
        return text
    words.insert(random.randint(0, len(words)), "absolutely")
    words.insert(random.randint(0, len(words)), "terrible")
    return " ".join(words)


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


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


st.set_page_config(page_title="Fake Review Detector", page_icon="🛡️", layout="centered")

model = load_model()

st.markdown("<h1 style='text-align:center;'>🛡️ Fake Review Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a review and the model will predict whether it is Real or Fake.</p>", unsafe_allow_html=True)

review = st.text_area(
    "Enter Review:",
    placeholder="Best best best product ever!!!",
    height=140
)

st.write("Optional attack testing:")

attack_option = st.radio(
    "",
    ["None", "Add Random Typo", "Inject Words", "Both Attacks", "Strong Attack"],
    horizontal=True
)

if st.button("Predict", use_container_width=True):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        processed_review = review

        if attack_option == "Add Random Typo":
            processed_review = add_typo(processed_review)
        elif attack_option == "Inject Words":
            processed_review = inject_words(processed_review)
        elif attack_option == "Both Attacks":
            processed_review = inject_words(add_typo(processed_review))
        elif attack_option == "Strong Attack":
            processed_review = strong_attack(processed_review)

        st.info(f"Processed Review: {processed_review}")

        cleaned = clean_text(processed_review)

        pred = model.predict([cleaned])[0]

        confidence = 0.80

        if hasattr(model, "decision_function"):
            score = model.decision_function([cleaned])[0]
            prob_real = 1 / (1 + np.exp(-score))
            confidence = prob_real if pred == 1 else 1 - prob_real

        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            confidence = max(probs)

        st.markdown("## Result")

        if pred == 0:
            st.error("Prediction: Fake ❌")
        else:
            st.success("Prediction: Real ✅")

        st.markdown("### Confidence Meter")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence * 100:.2f}%**")