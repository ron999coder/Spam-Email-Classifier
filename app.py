from flask import Flask, render_template, request, jsonify
import pickle
import os
from model.train import train_model

app = Flask(__name__)

MODEL_PATH = "model/spam_model.pkl"

# Load or train model on startup
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    train_model()

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "No text provided"}), 400

    prediction = pipeline.predict([email_text])[0]
    probability = pipeline.predict_proba([email_text])[0]

    label = "SPAM" if prediction == 1 else "NOT SPAM"
    confidence = round(float(max(probability)) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "is_spam": bool(prediction == 1)
    })


if __name__ == "__main__":
    app.run(debug=True)
