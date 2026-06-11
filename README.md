# 🛡️ Network Intrusion Detection System (Multi-Dataset ML System)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://network-intrusion-detection-122.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-style **machine learning system for network intrusion detection**, supporting multiple benchmark datasets (NSL-KDD + UNSW-NB15). The system performs multiclass attack classification, handles extreme class imbalance, and provides explainable predictions.

Designed as a portfolio project for ML engineering and security-focused AI roles.

---

## 🚀 Live Demo

![Demo](working_demo.gif)

👉 [Streamlit App](https://network-intrusion-detection-122.streamlit.app/)

Upload network traffic data (NSL-KDD format) to get:
- Real-time predictions
- Attack category classification
- Model confidence scores
- SHAP-based explanations (feature importance per prediction)

---

## 📊 Key Results

### 🔹 NSL-KDD Dataset

| Model | Macro F1 | CV Macro F1 | Accuracy |
|------|----------|-------------|----------|
| XGBoost | **0.5728** | 0.9349 ± 0.0272 | 79.4% |
| Random Forest | 0.4855 | 0.9034 ± 0.0471 | 74.1% |

---

### 🔹 UNSW-NB15 Dataset (Generalization Test)

| Model | Macro F1 | Accuracy |
|------|----------|----------|
| Random Forest | **0.6310** | 78% |
| XGBoost | 0.5855 | 76% |

---

### 📌 Why Macro F1?

The dataset is highly imbalanced. For example:
- Normal traffic dominates (~50%+ of samples)
- Rare attacks (R2L, U2R) are extremely underrepresented

Accuracy alone is misleading — Macro F1 ensures **equal weight to all attack classes**, making it the correct evaluation metric.

---

## 📌 Per-Class Performance (NSL-KDD — XGBoost)

| Class | Precision | Recall | F1-score |
|------|----------|--------|----------|
| DoS | 0.96 | 0.84 | 0.90 |
| Normal | 0.70 | 0.97 | 0.82 |
| Probe | 0.82 | 0.80 | 0.81 |
| R2L | 0.98 | 0.09 | 0.16 |
| U2R | 0.77 | 0.10 | 0.18 |

---

## 🔍 Key Insights

### 1. Dataset Shift Problem
A significant drop between cross-validation (~0.93) and test score (~0.57) is due to **distribution shift in NSL-KDD**, not overfitting.

The test set contains:
- Higher proportion of R2L attacks
- Different attack distribution than training data

This reflects a real-world security problem: **attack distributions evolve over time**.

---

### 2. Cross-Dataset Generalization (UNSW-NB15)

Adding UNSW-NB15 demonstrates:
- Model performance varies significantly across datasets
- Random Forest generalizes better on UNSW
- XGBoost performs better on NSL-KDD

👉 This confirms that intrusion detection is **dataset-dependent and non-stationary**

---

## 🏗️ System Architecture
             ┌─────────────────────┐
             │   NSL-KDD Dataset   │
             └─────────┬───────────┘
                       │
             ┌─────────▼───────────┐
             │   Preprocessing     │
             │ ─ Encoding          │
             │ ─ Scaling           │
             │ ─ Feature selection  │
             └─────────┬───────────┘
                       │
             ┌─────────▼───────────┐
             │  SMOTE Balancing    │
             └─────────┬───────────┘
                       │
    ┌──────────────────┴──────────────────┐
    ▼                                     ▼
    ┌─────────────────┐ ┌─────────────────┐
    │ Random Forest │ │ XGBoost │
    └────────┬────────┘ └────────┬────────┘
    │ │
    └────────────┬───────────────────┘
    ▼
    ┌────────────────────────────┐
    │ Evaluation (Macro F1, CV) │
    └────────────┬───────────────┘
    ▼
    ┌────────────────────────────┐
    │ SHAP Explainability Layer │
    └────────────┬───────────────┘
    ▼
    ┌────────────────────────────┐
    │ Streamlit Web Application │
    └────────────────────────────┘


---

## ⚙️ Technical Decisions

### ✔ Ordinal Encoding
Used for categorical features (protocol, service, flag) to avoid sparse explosion.

### ✔ SMOTE
Handles extreme class imbalance (e.g., U2R: <0.1% of data).

### ✔ Macro F1 Optimization
Ensures equal importance across all attack types.

### ✔ Cross-Dataset Validation
UNSW-NB15 added to test generalization beyond NSL-KDD.

### ✔ No Data Leakage in CV
SMOTE applied only on training set (not validation folds).

---

## 📁 Project Structure

```bash
nids-ml/
│
├── preprocess.py
├── train.py
├── explain.py
├── visualize.py
├── main.py
├── app.py
│
├── preprocess_unsw.py
├── run_unsw.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│
├── rf_model.pkl
├── rf_model_unsw.pkl
│
└── README.md

📦 Datasets
NSL-KDD
125,973 training samples
22,544 test samples
41 features
5 attack categories

https://www.unb.ca/cic/datasets/nsl.html

UNSW-NB15
175,341 training samples
82,332 test samples
Modern network attack dataset
Used for generalization testing

https://research.unsw.edu.au/projects/unsw-nb15-dataset

🚀 Quick Start
git clone https://github.com/your-repo/nids-ml.git
cd nids-ml

pip install -r requirements.txt

# Run NSL-KDD pipeline
python main.py

# Run UNSW-NB15 pipeline
python run_unsw.py

# Launch dashboard
streamlit run app.py

⚠️ Limitations
Low recall on rare attacks (R2L, U2R)
Dataset shift affects generalization
No live network packet ingestion yet
No continuous retraining pipeline
🔮 Future Work
Real-time packet capture integration (Wireshark/Scapy)
Streaming inference pipeline
Continual learning system
Model calibration for rare attacks
Deployment as API service
📜 License

MIT License