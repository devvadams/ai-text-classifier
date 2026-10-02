# 🧠 AI Text Classifier API

A simple and production-ready machine learning API that classifies text messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP).

---

## 🚀 Features

* Spam detection using Machine Learning
* TF-IDF vectorization with n-grams
* Logistic Regression model (with class balancing)
* REST API built with Flask
* Confidence score for predictions
* Metrics endpoint for model evaluation
* Uncertainty handling for ambiguous inputs

---

## 📂 Project Structure

```
ai-text-classifier/
│
├── app/
│   ├── routes.py
│   └── services/
│       └── model_service.py
│
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── config.py
├── run.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/devvadams/ai-text-classifier/ai-text-classifier-v2.git
cd ai-text-classifier
pip install -r requirements.txt
```

---

## 🧪 Train the Model

```bash
python -m ml.train
```

---

## ▶️ Run the API

```bash
python run.py
```

Server runs on:

```
http://127.0.0.1:5000
```

---

## 📡 API Endpoints

### 1. Predict Spam

**POST /predict**

```json
{
  "text": "You have won a free iPhone"
}
```

**Response:**

```json
{
  "prediction": "spam",
  "confidence": 0.97
}
```

---

### 2. Model Metrics

**GET /metrics**

```json
{
  "accuracy": 0.96,
  "samples": 5157
}
```

---

### 3. Health Check

**GET /health**

```json
{
  "status": "ok"
}
```

---

## 🧠 How It Works

1. Text is cleaned and preprocessed
2. Converted into numerical features using TF-IDF
3. Model predicts class (spam/ham)
4. Confidence score is returned
5. Low-confidence predictions can be marked as **uncertain**

---

## 📌 Notes

* Dataset and trained models are excluded from the repository
* Run the training script to generate models locally

---

## 💼 Author

**Adamu Baba Hassan**
Backend Engineer transitioning into AI/ML Engineer

GitHub: https://github.com/devvadams

---

## ⭐ Acknowledgment
* If you found this repo useful give me a start ⭐
Dataset: UCI SMS Spam Collection
