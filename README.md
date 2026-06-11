# 🛡️ Network Intrusion Detection System (ML-Based)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://network-intrusion-detection-122.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A production-style machine learning system that classifies network traffic into attack categories in real time, with explainable predictions via SHAP. Built as a portfolio project targeting ML engineering roles.

**[→ Live Demo](https://network-intrusion-detection-122.streamlit.app/)**

---

## 📊 Demo

Upload any NSL-KDD formatted CSV to get live intrusion predictions with per-record SHAP explanations.

*(Add your demo.gif here once recorded)*

---

## 📈 Key Results

| Model | Test Macro F1 | CV Macro F1 | Accuracy |
|------|--------------|--------------|----------|
| XGBoost | **0.5728** | 0.9349 ± 0.0272 | 79.4% |
| Random Forest | 0.4855 | 0.9034 ± 0.0471 | 74.1% |

### Why Macro F1 and not accuracy?

The dataset is heavily imbalanced — Normal traffic accounts for ~53% of test samples. A model predicting only Normal would still get ~53% accuracy while missing all attacks. Macro F1 treats all 5 classes equally and reflects true performance.

---

### 📌 Per-class breakdown (XGBoost)

| Class | Precision | Recall | F1 |
|------|----------|--------|----|
| DoS | 0.96 | 0.84 | 0.90 |
| Normal | 0.70 | 0.97 | 0.82 |
| Probe | 0.82 | 0.80 | 0.81 |
| R2L | 0.98 | 0.09 | 0.16 |
| U2R | 0.77 | 0.10 | 0.18 |

---

## 🔍 Key Finding — Train/Test Distribution Shift

The large gap between CV score (~0.93) and test score (~0.57) is not overfitting. It is a known property of the NSL-KDD dataset: the test set (`KDDTest+`) has a different attack distribution than the training set.

Specifically, R2L attacks increase from ~0.8% in training to ~12% in testing. The model has limited exposure to these patterns, leading to poor generalization on rare attack types.

This reflects a real-world intrusion detection challenge: attack distributions evolve over time. Addressing this would require continual learning or periodic retraining, noted as future work.

---

## 🏗️ Architecture
NSL-KDD Dataset
│
├── preprocess.py
│ ├── OrdinalEncoder (protocol_type, service, flag)
│ ├── StandardScaler (numeric features)
│ └── ColumnTransformer (fit on train only)
│
├── train.py
│ ├── SMOTE (class balancing)
│ ├── Random Forest
│ └── XGBoost
│
├── explain.py
│ ├── SHAP TreeExplainer
│ ├── Global beeswarm plots
│ └── Local waterfall explanations
│
└── app.py (Streamlit)
├── CSV upload
├── Live predictions
├── Confidence scores
└── SHAP explanations


---

## ⚙️ Technical Decisions

### Why OrdinalEncoder?
Avoids incorrect ordering assumptions from LabelEncoder and safely handles unseen categories.

### Why SMOTE?
Balances extreme class imbalance (e.g., R2L: 995 vs Normal: 67,343) by generating synthetic minority samples.

### Why Macro F1?
Ensures equal importance for all attack classes, especially rare but critical ones (R2L, U2R).

### Why avoid SMOTE before CV?
Prevents data leakage. SMOTE is applied only on training data, while CV is performed on original distribution.

---

## 📁 Project Structure
nids-ml/
├── preprocess.py
├── train.py
├── explain.py
├── visualize.py
├── main.py
├── app.py
├── requirements.txt
├── results/
│ ├── confusion_matrix.png
│ ├── feature_importance.png
│ ├── shap_summary.png
│ └── shap_waterfall_*.png
└── README.md


---

## 🚀 Quick Start

```bash
git clone https://github.com/YOURUSERNAME/nids-ml.git
cd nids-ml

pip install -r requirements.txt

# Download dataset
# Place KDDTrain+.txt and KDDTest+.txt in root

python main.py
python explain.py
streamlit run app.py

📦 Dataset

NSL-KDD — University of New Brunswick
125,973 training samples | 22,544 test samples
41 features | 5 attack categories

https://www.unb.ca/cic/datasets/nsl.html

⚠️ Limitations & Future Work
R2L and U2R have low recall due to dataset distribution shift
Only evaluated on NSL-KDD (no cross-dataset validation yet)
No live network packet ingestion pipeline
No continuous retraining mechanism
📜 License

MIT