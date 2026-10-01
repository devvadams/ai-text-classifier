# 🚀 AI Text Classifier API

A simple yet well-structured **Flask-based REST API** that uses **Machine Learning (Scikit-learn)** to classify text as **Spam** or **Not Spam**.

This project demonstrates backend API development, data processing, and machine learning integration in a clean, modular architecture.

---

## 📌 Features

* RESTful API built with Flask
* Text classification using Machine Learning
* NLP preprocessing with CountVectorizer
* Model persistence using Joblib (no retraining on every run)
* Modular project structure (routes, model, data separation)

---

## 🧠 How It Works

1. A request is sent to the `/predict` endpoint with text input
2. The text is transformed using a trained vectorizer
3. The ML model predicts whether the text is spam or not
4. A JSON response is returned

---

## 🛠️ Tech Stack

* **Python**
* **Flask**
* **Pandas**
* **Scikit-learn**
* **Joblib**

---

## 📂 Project Structure

```
ai-text-classifier/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── model.py
│   └── models/
│       ├── model.pkl
│       └── vectorizer.pkl
│
├── data/
│   └── dataset.py
│
├── run.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/ai-text-classifier.git
cd ai-text-classifier
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```
python run.py
```

Server will start at:

```
http://127.0.0.1:5000/
```

---

## 🧪 API Usage

### Endpoint

```
POST /predict
```

### Request Body (JSON)

```
{
  "text": "You have won a free iPhone"
}
```

### Response

```
{
  "input": "You have won a free iPhone",
  "prediction": "spam"
}
```

---

## 📈 Example Use Cases

* Spam detection systems
* Email filtering
* Message moderation tools
* NLP-based backend services

---

## 🔥 Key Highlights

* Clean and modular backend architecture
* Integration of machine learning into a REST API
* Efficient model loading using Joblib
* Beginner-friendly yet production-inspired structure

---

## 🚧 Future Improvements

* Add more training data for better accuracy
* Replace CountVectorizer with TF-IDF
* Integrate deep learning (e.g., Transformers)
* Add authentication (JWT)
* Deploy to cloud (Render / Railway)

---

## 👨‍💻 Author

**Adamu Baba Hassan**
Backend Engineer | Python & AI Enthusiast

---

## 📬 Contact

* Email: [babaadamu2019@gmail.com](mailto:babaadamu2019@gmail.com)
* LinkedIn: https://linkedin.com/in/adamu-babahassan-750563248

---

## ⭐ If you found this useful

Give the repo a star ⭐ and feel free to contribute!
