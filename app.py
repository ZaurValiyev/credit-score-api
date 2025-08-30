from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load pipeline
with open("credit_score_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    df = pd.read_csv(file)

    preds = model.predict(df)
    return jsonify({"predictions": preds.tolist()})

@app.route("/")
def home():
    return "✅ Credit Score API is running!"
