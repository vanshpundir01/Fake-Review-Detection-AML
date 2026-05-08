# 🍔 Fake Review Detection using Adversarial Machine Learning

> Detect fake food reviews using NLP, Machine Learning, and Adversarial AI techniques.

---

## 📌 Overview

Online food platforms contain thousands of reviews that influence customer decisions every day.  
However, many reviews are manipulated, spammed, or artificially generated to boost ratings.

This project uses **Natural Language Processing (NLP)** and **Adversarial Machine Learning (AML)** to classify reviews as:

✅ **Real Reviews**  
❌ **Fake Reviews**

The project also demonstrates how attackers can manipulate reviews using adversarial attacks and how defensive training improves robustness.

---

# 🍕 Features

✅ Fake vs Real Food Review Detection  
✅ Beautiful Streamlit UI  
✅ Confidence Meter  
✅ Adversarial Attack Simulation  
✅ Defense Model Training  
✅ NLP Text Processing  
✅ TF-IDF Feature Extraction  
✅ Real-time Prediction System  

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Scikit-learn | Machine Learning |
| NLP | Text Processing |
| TF-IDF | Feature Extraction |
| Streamlit | Web Application |
| Pandas | Data Handling |
| NumPy | Numerical Operations |

---

# ⚔️ Adversarial Attacks

This project demonstrates multiple attacks on review classification systems:

- 🔤 Random Typo Attack
- 💬 Word Injection Attack
- 🧨 Strong Semantic Attack
- 🔀 Combined Adversarial Attack

These attacks attempt to fool the model into incorrect predictions.

---

# 🛡️ Defense Mechanism

A separate defense model is trained using:
- Clean reviews
- Adversarially modified reviews

This improves the robustness of the fake review detector against attacks.

---

# 📊 Model Performance

| Model | Accuracy |
|---|---|
| Standard Review Detector | ~89% |
| Under Adversarial Attack | ~86% |
| Defense Model | Improved Robustness |

---

# 🍟 Project Structure

```text
fake_review_project/
│
├── train.py
├── test.py
├── attack_test.py
├── train_defense.py
├── test_defense.py
├── app.py
│
├── model.pkl
├── defense_model.pkl
│
├── train_data.csv
├── test_data.csv
├── new_data_test.csv
│
├── requirements.txt
├── README.md
└── screenshots/