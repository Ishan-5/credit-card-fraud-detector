# Credit Card Fraud Detector

A production-ready machine learning pipeline that detects fraudulent credit card transactions with 93% precision and 86% recall, deployed as a REST API with a live web dashboard.

**Live Demo:** https://credit-card-fraud-detector-5dh4.onrender.com

---

## Overview

Credit card fraud detection is a classic imbalanced classification problem — only 0.17% of transactions are fraudulent. This project builds a full ML pipeline from raw data to a deployed API, making deliberate decisions at each step to handle the class imbalance correctly.

---

## Results

| Metric | Value |
|---|---|
| F1 Score (fraud class) | 0.89 |
| ROC-AUC | 0.963 |
| Precision | 0.93 |
| Recall | 0.86 |
| Frauds caught (test set) | 84 / 98 |
| False alarms | 6 / 56,864 |
| Decision threshold | 0.41 (tuned) |

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Transactions made by European cardholders in September 2013. 284,807 transactions, 492 frauds (0.172%). Features V1–V28 are PCA-transformed by the original authors. Amount and Time are the only raw features.

---

## Key ML Decisions

**Why not accuracy?**
Predicting every transaction as legit gives 99.83% accuracy. Meaningless for fraud. The right metrics are Precision, Recall, F1, and Average Precision (area under PR curve).

**Why no SMOTE?**
SMOTE was tested and hurt recall (0.86 → 0.78). The Random Forest was already powerful enough to learn from raw imbalanced data. Adding synthetic samples introduced noise.

**Why threshold tuning?**
Default threshold of 0.5 gave F1=0.87. Tuning to 0.41 improved F1 to 0.89 by catching 4 more frauds at the cost of 1 extra false alarm — a good trade in any fraud system.

**Why Random Forest over XGBoost?**
Both performed similarly (RF: F1=0.87, XGB: F1=0.86). Random Forest was chosen for its simplicity and interpretability.

---

## Project Structure

```
credit-card-fraud-detector/
├── app.py                  ← Flask REST API
├── templates/
│   └── index.html          ← Frontend dashboard
├── models/
│   ├── best_model.pkl      ← Trained Random Forest
│   ├── scaler.pkl          ← StandardScaler for Amount & Time
│   └── metadata.json       ← Threshold and feature names
├── requirements.txt
└── README.md
```

---

## API

### `GET /health`
```json
{
  "status": "ok",
  "model": "RandomForest",
  "threshold": 0.41
}
```

### `POST /predict`
Send a transaction as JSON with all 30 features (V1–V28, Amount, Time).

```bash
curl -X POST https://credit-card-fraud-detector-5dh4.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.36, "V2": -0.07, ..., "Amount": 149.62, "Time": 0}'
```

Response:
```json
{
  "fraud_probability": 0.03,
  "is_fraud": false,
  "risk_level": "LOW"
}
```

Risk levels: `LOW` (< 0.4), `MEDIUM` (0.4–0.7), `HIGH` (> 0.7)

---

## Run Locally

```bash
git clone https://github.com/Ishan-5/credit-card-fraud-detector
cd credit-card-fraud-detector
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`

---

## Tech Stack

- scikit-learn — Random Forest, StandardScaler, metrics
- Flask + Flask-CORS — REST API
- pandas, numpy — data processing
- matplotlib, seaborn — EDA visualizations

---

## Author

**Devansh Kumar Pandey**  
[GitHub](https://github.com/Ishan-5)
