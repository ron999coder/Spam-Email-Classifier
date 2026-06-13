# 📧 Spam Email Classifier

An end-to-end NLP-based spam detection web app built with Python, Scikit-learn, NLTK, and Flask.

## 🔧 Tech Stack
- **ML:** Scikit-learn (Multinomial Naive Bayes + TF-IDF)
- **NLP:** NLTK (stopword removal, text normalization)
- **Backend:** Flask (REST API)
- **Frontend:** HTML, CSS, JavaScript (Fetch API)

## 📁 Project Structure
```
spam-classifier/
├── app.py                  # Flask app & API routes
├── requirements.txt
├── model/
│   ├── train.py            # Preprocessing + model training pipeline
│   └── spam_model.pkl      # Saved trained model (auto-generated)
├── templates/
│   └── index.html          # UI
└── static/
    └── css/style.css
```

## 🚀 Setup & Run

```bash
# 1. Clone / download the project
cd spam-classifier

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app (model trains automatically on first run)
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## 🧠 How It Works

1. **Preprocessing:** Email text is lowercased, URLs and numbers are normalized, stopwords are removed using NLTK.
2. **Feature Extraction:** TF-IDF Vectorizer with bigrams converts cleaned text into numerical features.
3. **Model:** Multinomial Naive Bayes classifies the feature vector as spam (1) or ham (0).
4. **API:** Flask exposes a `/predict` POST endpoint that returns the label and confidence score.

## 📊 Model Details
- Algorithm: Multinomial Naive Bayes
- Features: TF-IDF (unigrams + bigrams, 5000 max features, sublinear TF scaling)
- Preprocessing: Lowercasing, URL/number normalization, NLTK stopword removal

## 📌 For a Real Dataset
Replace `get_training_data()` in `model/train.py` with the **SMS Spam Collection Dataset**:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

```python
import pandas as pd
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
texts = df["text"].tolist()
labels = (df["label"] == "spam").astype(int).tolist()
```

---

## ✅ Resume Bullet Points

> - Built an **NLP-based Spam Email Classifier** using Python, Scikit-learn (Naive Bayes + TF-IDF), and NLTK; deployed via a Flask REST API with a responsive web UI
> - Implemented a full ML pipeline — text preprocessing, feature extraction, model training, and serialization with Pickle
> - Achieved real-time predictions with confidence scoring through a `/predict` API endpoint integrated with a vanilla JS frontend

---

Built by **Ranojoy Saha** — B.Tech CSE (AI/ML), Techno International New Town
GitHub: [github.com/ron999coder](https://github.com/ron999coder)
