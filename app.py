from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify
import joblib, json
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model, scaler, metadata
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
metadata = json.load(open('models/metadata.json'))
THRESHOLD = metadata['threshold']
FEATURES = metadata['features']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "RandomForest", "threshold": THRESHOLD})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    df = pd.DataFrame([data], columns=FEATURES)
    
    # Scale using values only (no column name checking)
    df['Amount'] = scaler.transform(df[['Amount']].values)
    df['Time'] = scaler.transform(df[['Time']].values)
    
    prob = float(model.predict_proba(df)[0][1])
    is_fraud = prob >= THRESHOLD
    
    return jsonify({
        "fraud_probability": round(prob, 6),
        "is_fraud": is_fraud,
        "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)