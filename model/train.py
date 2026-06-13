import pickle
import os
import re
import nltk
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data (only first time)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Clean and normalize email text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)      # URLs
    text = re.sub(r"\b\d+\b", " num ", text)              # Numbers
    text = re.sub(r"[^a-z\s]", " ", text)                 # Non-alpha
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def get_training_data():
    """
    Returns (texts, labels).
    Labels: 1 = spam, 0 = ham.

    In a real project, load the SMS Spam Collection dataset:
    https://archive.ics.uci.edu/dataset/228/sms+spam+collection
    Here we use a representative built-in sample for demo purposes.
    """
    spam_samples = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now!",
        "URGENT: Your account has been suspended. Verify immediately to avoid charges.",
        "FREE entry in our prize draw! Text WIN to 80182 now! T&C apply.",
        "You have been selected for a cash prize of $500. Call us now to collect.",
        "Limited offer! Buy 1 get 1 FREE on all products. Shop now!",
        "Your loan has been approved! Get $5000 deposited today. Click here.",
        "WINNER!! You are selected as our lucky winner. Claim your prize now.",
        "Earn money from home! $500/day guaranteed. No experience needed.",
        "Alert: Your card was charged $499. Call 1800-xxx-xxxx to dispute.",
        "You've been pre-approved for a credit card with 0% interest! Apply now.",
        "Hot singles in your area! Click now for free access tonight.",
        "Cheap meds online! No prescription needed. Order now and save 80%.",
        "Investment opportunity of a lifetime! Guaranteed 300% return. Act now!",
        "Your computer has a virus! Download our free scanner immediately.",
        "Congratulations, you're the 1,000,000th visitor! Claim your iPhone now.",
        "Make $3000/week from home. Proven system. Start today. Click here.",
        "Final notice: Renew your subscription or lose access permanently.",
        "Exclusive deal: Rolex watches at 90% off. Order before midnight!",
        "Your PayPal account will be closed. Update your details now.",
        "Win a luxury cruise for two! Limited slots available. Register now!",
    ]

    ham_samples = [
        "Hey, are we still meeting at 3pm today? Let me know if something came up.",
        "Can you please send me the report by end of day? Thanks!",
        "Just a reminder that your dentist appointment is scheduled for Friday at 10am.",
        "The project deadline has been moved to next Monday. Please plan accordingly.",
        "Mom said dinner is ready. Come home soon.",
        "I'll be a bit late to the meeting. Please start without me.",
        "Your order #12345 has been shipped and will arrive in 2-3 business days.",
        "Happy birthday! Hope you have a great day.",
        "The lecture notes for today's class have been uploaded on the portal.",
        "Can you review my code before I push it to main? I need a second opinion.",
        "Library books you borrowed are due for return this Friday.",
        "Team standup is at 10am tomorrow. Please be on time.",
        "The electricity bill for this month is Rs. 1,200. Kindly pay by 20th.",
        "Your flight booking is confirmed. Check-in opens 24 hours before departure.",
        "We're having a potluck lunch tomorrow. Please bring something vegetarian.",
        "Sorry I missed your call. I'll call you back in an hour.",
        "The assignment submission portal is now open. Deadline is Sunday 11:59pm.",
        "Monthly review meeting is scheduled for next Tuesday at 2pm.",
        "Your hostel fee payment receipt has been sent to your registered email.",
        "Just checking in — how are you doing? Haven't heard from you in a while.",
    ]

    texts = spam_samples + ham_samples
    labels = [1] * len(spam_samples) + [0] * len(ham_samples)
    return texts, labels


def train_model():
    """Train the spam classifier and save it to disk."""
    texts, labels = get_training_data()
    processed = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    os.makedirs("model", exist_ok=True)
    with open("model/spam_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Model saved to model/spam_model.pkl")
    return pipeline


if __name__ == "__main__":
    train_model()
