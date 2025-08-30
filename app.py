from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# --- Load the trained pipeline ---
pipeline = joblib.load("credit_score_pipeline.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    df = pd.read_csv(file)

    # --- Make predictions ---
    preds = pipeline.predict(df)
    return jsonify({"predictions": preds.tolist()})

@app.route("/")
def home():
    return "✅ Credit Score API is running!"

if __name__ == "__main__":
    app.run(debug=True)
