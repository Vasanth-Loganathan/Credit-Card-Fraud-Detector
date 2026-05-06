# 💳 Credit Card Fraud Detection

> **An end-to-end machine learning pipeline for real-time credit card fraud detection using the Sparkov synthetic dataset — featuring a weighted gradient boosting ensemble, SHAP explainability, FP-Growth association rule mining, and a live Streamlit web application.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-0.9506_AUPRC-orange"/>
  <img src="https://img.shields.io/badge/Ensemble-0.9517_AUPRC-brightgreen"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet"/>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Stage 1 — Data Cleaning](#stage-1--data-cleaning)
  - [Stage 2 — Feature Engineering](#stage-2--feature-engineering)
  - [Stage 3 — Encoding & Transforms](#stage-3--encoding--transforms)
  - [Stage 4 — Train-Test Split & Scaling](#stage-4--train-test-split--scaling)
  - [Stage 5 — SMOTE Oversampling](#stage-5--smote-oversampling)
- [Association Rule Mining — FP-Growth](#-association-rule-mining--fp-growth)
- [Models Trained](#-models-trained)
- [Model Performance](#-model-performance)
- [Threshold Optimization](#-threshold-optimization)
- [SHAP Explainability](#-shap-explainability)
- [Streamlit App — FraudGuard](#-streamlit-app--fraudguard)
- [Installation & Usage](#-installation--usage)

---

## 🔍 Overview

Credit card fraud is a global financial problem costing billions annually. This project builds a **production-ready, explainable fraud detection system** on the Sparkov synthetic transaction dataset. The pipeline handles extreme class imbalance (191:1), engineers meaningful geospatial and temporal features from raw transaction data, mines fraud patterns using FP-Growth, trains and compares **9 models** across 5 algorithm families, and deploys the champion ensemble model as an interactive web dashboard.

**Key results from the notebook:**
- 🏆 **AUPRC of 0.9517** with a Weighted Ensemble (XGBoost × 3 + LightGBM × 1)
- 🥇 **XGBoost alone achieves AUPRC 0.9506** — Precision 94%, Recall 87%
- ⚖️ Solved **191:1 class imbalance** using SMOTE (7,721 → 294,838 fraud samples in training)
- 🔗 **FP-Growth** discovered top rule: *Male + Very High Amount + Shopping (Online) + Evening → Fraud* with **97% confidence and 10.23× lift**
- 🔎 **SHAP** explains every fraud prediction feature-by-feature using TreeExplainer
- 🌐 **Streamlit dashboard** for real-time transaction analysis with live SHAP waterfall charts

---

## ❗ Problem Statement

| Challenge | Detail |
|-----------|--------|
| **Extreme Class Imbalance** | Only 0.58% of transactions are fraudulent (1,930 fraud in 370,479 test records) |
| **Misleading Accuracy** | A naive "always predict Legitimate" model achieves 99.4% accuracy yet catches zero fraud |
| **Correct Metric** | AUPRC (Area Under Precision-Recall Curve) — penalises missing fraud AND false alarms |
| **Explainability** | Banks must justify fraud block decisions; SHAP provides per-transaction explanations |

---

## 📊 Dataset

**Source:** [Sparkov Synthetic Credit Card Transaction Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

| Property | Value |
|----------|-------|
| Total Transactions | 1,852,394 |
| Raw Features | 23 columns |
| Engineered Features | 24 features (after pipeline) |
| Fraud Transactions | ~10,700 (0.58%) |
| Class Imbalance Ratio | ~191 : 1 |
| Training Records | 1,481,915 |
| Test Records | 370,479 (stratified split) |

**Raw columns used:**

| Column | Type | Description |
|--------|------|-------------|
| `trans_date_trans_time` | Datetime | Exact transaction timestamp |
| `amt` | Float | Transaction amount (USD) |
| `category` | String | Merchant category (14 types) |
| `lat` / `long` | Float | Customer GPS coordinates |
| `merch_lat` / `merch_long` | Float | Merchant GPS coordinates |
| `city_pop` | Int | Population of cardholder's city |
| `dob` | Date | Customer date of birth |
| `merchant`, `job`, `city` | String | High-cardinality categoricals |
| `gender` | String | Customer gender (M/F) |
| `is_fraud` | Int | Target label (0=Legitimate, 1=Fraud) |

**Dropped columns (9):** `Unnamed: 0`, `cc_num`, `trans_num`, `unix_time`, `first`, `last`, `street`, `state`, `zip`

---

## 📁 Project Structure

```
credit-card-fraud-detector/
│
├── Models/
│   ├── ensemble_model.pkl      ← Champion: Weighted Ensemble (XGBoost×3 + LightGBM×1)
│   ├── xgb_model.pkl           ← XGBoost (AUPRC 0.9506)
│   ├── lgbm_model.pkl          ← LightGBM (AUPRC 0.8425)
│   ├── rf_model.pkl            ← Random Forest (AUPRC 0.8172)
│   ├── nn_model.pkl            ← Keras Neural Network (AUPRC ~0.89)
│   ├── dt_model.pkl            ← Decision Tree (AUPRC 0.8637)
│   ├── scaler.pkl              ← Fitted StandardScaler (fit on X_train only)
│   ├── metadata.pkl            ← Feature names, freq medians, optimal threshold
│   ├── dashboard_metrics.pkl   ← Pre-computed leaderboard metrics for app
│   ├── shap_data.pkl           ← Pre-computed SHAP values (2000-sample subset)
│   └── fraud_rules.csv         ← Association rules from FP-Growth
│
├── app.py                      ← Streamlit web app (FraudGuard dashboard)
├── Notebook.ipynb              ← Full ML pipeline: EDA → Training → SHAP → Save
├── requirements.txt            ← Python dependencies
└── README.md
```

---

## 🤖 Machine Learning Pipeline

The raw data goes through a **5-stage preprocessing pipeline** before any model sees it. Each stage output is verified with shape checks in the notebook.

```
Raw CSV (1,852,394 × 23)
    ↓ Stage 1 — Cleaning
(1,852,394 × 14)
    ↓ Stage 2 — Feature Engineering
(1,852,394 × 13)
    ↓ Stage 3 — Encoding & Log Transform
(1,852,394 × 25)
    ↓ Stage 4 — Train/Test Split + StandardScaler
Train: 1,481,915 rows   Test: 370,479 rows (24 features each)
    ↓ Stage 5 — SMOTE (train only)
Train: 1,769,032 rows (Legit: 1,474,194 | Fraud: 294,838)
```

### Stage 1 — Data Cleaning

```python
cols_to_drop = ['Unnamed: 0', 'cc_num', 'trans_num', 'unix_time',
                'first', 'last', 'street', 'state', 'zip']
df.drop(columns=cols_to_drop, inplace=True)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])
# Shape: (1,852,394, 14)
```

### Stage 2 — Feature Engineering

| New Feature | Source | Fraud Signal |
|-------------|--------|-------------|
| `hour` | `trans_date_trans_time.dt.hour` | Fraud spikes 12 AM – 6 AM |
| `day_of_week` | `trans_date_trans_time.dt.dayofweek` | Weekend patterns differ |
| `month` | `trans_date_trans_time.dt.month` | Seasonal fraud spikes |
| `age` | `trans_year - dob.year` | Seniors (65+) disproportionately targeted |
| `distance_km` | Haversine(lat/long, merch_lat/long) | Far-from-home transactions signal fraud |

**Haversine Distance Formula (vectorized with NumPy):**
```python
def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))  # km
```

After feature engineering, `trans_date_trans_time`, `dob`, `lat`, `long`, `merch_lat`, `merch_long` are dropped. Shape becomes **(1,852,394 × 13)**.

### Stage 3 — Encoding & Transforms

```python
# Log Transform (reduces right-skew in transaction amounts)
df['amt_log'] = np.log1p(df['amt'])
df.drop(columns=['amt'], inplace=True)

# Frequency Encoding (high-cardinality: merchant, job, city)
for col in ['merchant', 'job', 'city']:
    freq = df[col].value_counts() / len(df)
    df[col + '_freq'] = df[col].map(freq)
    df.drop(columns=[col], inplace=True)

# One-Hot Encoding (drop_first=True prevents dummy variable trap)
df = pd.get_dummies(df, columns=['category', 'gender'], drop_first=True)
# Shape: (1,852,394, 25)  — 24 features + 1 target
```

**Final 24 features:** `city_pop`, `hour`, `day_of_week`, `month`, `age`, `distance_km`, `amt_log`, `merchant_freq`, `job_freq`, `city_freq`, `category_food_dining`, `category_gas_transport`, `category_grocery_net`, `category_grocery_pos`, `category_health_fitness`, `category_home`, `category_kids_pets`, `category_misc_net`, `category_misc_pos`, `category_personal_care`, `category_shopping_net`, `category_shopping_pos`, `category_travel`, `gender_M`

### Stage 4 — Train-Test Split & Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Training: 1,481,915 rows  |  Test: 370,479 rows

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)        # Transform only — no leakage
```

### Stage 5 — SMOTE Oversampling

```python
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```

| | Before SMOTE | After SMOTE |
|-|:-----------:|:-----------:|
| Legitimate | 1,474,194 | 1,474,194 |
| **Fraud** | **7,721** | **294,838** |
| Total | 1,481,915 | 1,769,032 |

> ⚠️ SMOTE is applied **only to training data**. The test set (370,479 rows) is never touched — this prevents optimistic bias in evaluation.

---

## 🔗 Association Rule Mining — FP-Growth

FP-Growth is run on all fraud cases + 5% random sample of legitimate cases. Continuous features are binned into categorical intervals before mining.

**Binning strategy:**
```python
df_fast['dist_cat']    = pd.qcut(distance_km, q=4,
    labels=['Dist_Very_Close','Dist_Close','Dist_Far','Dist_Very_Far'])
df_fast['age_cat']     = pd.qcut(age, q=4,
    labels=['Age_Young','Age_Adult','Age_Middle','Age_Senior'])
df_fast['amount_cat']  = pd.qcut(amt_log, q=4,
    labels=['Amt_Low','Amt_Med','Amt_High','Amt_Very_High'])
df_fast['time_of_day'] = pd.cut(hour, bins=[-1,6,12,18,24],
    labels=['Time_Night','Time_Morning','Time_Afternoon','Time_Evening'])

frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
```

**Top 5 Discovered Fraud Rules (sorted by Lift):**

| Antecedents (If these conditions hold…) | Confidence | Lift |
|-----------------------------------------|:----------:|:----:|
| Male + Amt_Very_High + shopping_net + Time_Evening | 0.9704 | **10.23×** |
| Amt_Very_High + shopping_net + Time_Evening | 0.8756 | 9.24× |
| Male + shopping_net + Time_Evening | 0.7163 | 7.55× |
| Age_Senior + Amt_Very_High + Time_Evening | 0.7124 | 7.51× |
| Male + Amt_Very_High + shopping_net | 0.6808 | 7.18× |

**Interpretation of the top rule:** When a male cardholder makes a very high-value online shopping transaction in the evening, the model assigns fraud probability 97% — that is **10.23× higher** than the baseline fraud rate.

---

## 🧠 Models Trained

| # | Model | Algorithm Family | Key Configuration |
|---|-------|-----------------|-------------------|
| 1 | Decision Tree | Tree | `max_depth=10`, Gini impurity, trained on SMOTE data |
| 2 | SGD Logistic Regression | Linear | `loss='log_loss'`, L2 regularization |
| 3 | Naive Bayes | Probabilistic | GaussianNB + Yeo-Johnson PowerTransformer pipeline |
| 4 | K-Nearest Neighbors | Distance-Based | `k=5`, trained on 10% stratified sample |
| 5 | Neural Network | Deep Learning | Dense(64)→BN→Drop(0.4)→Dense(32)→BN→Drop(0.4)→Sigmoid |
| 6 | Random Forest | Ensemble Bagging | 100 trees, `max_depth=10`, SMOTE data |
| 7 | XGBoost | Gradient Boosting | 300 trees, `tree_method='hist'`, CUDA GPU, SMOTE data |
| 8 | LightGBM | Gradient Boosting | 2000 estimators, `scale_pos_weight=190.9`, no SMOTE |
| 9 | Isolation Forest | Unsupervised Anomaly | 500 trees, `contamination=0.02`, trained on raw X_train |
| 🏆 | **Weighted Ensemble** | **Custom Voting** | **XGBoost (weight=3) + LightGBM (weight=1), soft voting** |

**Neural Network Architecture (Keras/TensorFlow):**
```
Input (24 features)
  → Dense(64, relu) → BatchNormalization → Dropout(0.4)
  → Dense(32, relu) → BatchNormalization → Dropout(0.4)
  → Dense(1, sigmoid)

Optimizer : Adam(lr=0.001)
Loss      : binary_crossentropy
Metric    : PR-AUC (keras.metrics.AUC(curve='PR'))
Callbacks : EarlyStopping(monitor='val_pr_auc', patience=5, restore_best_weights=True)
Batch     : 2048  |  Max epochs: 50
```

**LightGBM Configuration:**
```python
lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.01,
    max_depth=10, num_leaves=150,
    min_data_in_leaf=50,
    scale_pos_weight=190.9,   # Computed from actual class ratio
    callbacks=[early_stopping(stopping_rounds=100)]
)
# Best iteration: [16]  — early stopped very quickly
# Optimal threshold: 0.1713 (PR-curve optimized)
```

---

## 📈 Model Performance

All metrics are measured on the **held-out test set (370,479 rows, never seen during training)**.

| Rank | Model | AUPRC | Precision | Recall | F1-Score |
|:----:|-------|:-----:|:---------:|:------:|:--------:|
| 🥇 | **Weighted Ensemble (XGB+LGBM)** | **0.9517** | **0.93** | **0.87** | **0.90** |
| 🥈 | XGBoost | 0.9506 | 0.94 | 0.87 | 0.90 |
| 🥉 | Decision Tree | 0.8637 | 0.40 | 0.92 | 0.56 |
| 4 | Neural Network | ~0.8894 | — | — | — |
| 5 | Random Forest | 0.8172 | 0.63 | 0.79 | 0.70 |
| 6 | LightGBM | 0.8425 | 0.87 | 0.81 | 0.84 |
| 7 | KNN (10% sample) | 0.3478 | 0.19 | 0.75 | 0.30 |
| 8 | SGD Logistic Reg. | 0.2474 | 0.15 | 0.58 | 0.24 |
| 9 | Naive Bayes | 0.2248 | 0.01 | 0.74 | 0.03 |
| 10 | Isolation Forest | 0.0162 | 0.02 | 0.08 | 0.03 |

> **Why AUPRC?** With 191:1 class imbalance, accuracy is trivially 99.4% even for a model that catches zero fraud. AUPRC measures the trade-off between Precision (avoiding false alarms) and Recall (catching actual fraud) across all threshold values — it is the standard metric for imbalanced fraud detection.

> **Note on Isolation Forest:** Being fully unsupervised (no labels during training), it achieves poor AUPRC (0.0162) on this dataset. Its value in the system is detecting novel, previously unseen fraud patterns that supervised models haven't been trained on.

> **Note on Decision Tree:** High Recall (0.92) but very low Precision (0.40) — it catches almost all fraud but generates too many false alarms. Useful as a high-sensitivity filter, not as a standalone decision maker.

---

## 🎯 Threshold Optimization

XGBoost outputs a continuous fraud probability [0.0, 1.0]. The default 0.5 cutoff is sub-optimal for fraud detection. The notebook scans the full Precision-Recall curve to find the F1-maximizing threshold:

```
============================================================
THRESHOLD OPTIMIZATION — XGBoost
============================================================

Default threshold (0.50):
  Recall:    0.8705
  Precision: 0.9375

Optimal F1 threshold (0.582):
  Recall:    0.8575
  Precision: 0.9572
  F1-Score:  0.9046

Saved optimal threshold: 0.5819
```

The threshold `0.5819` is serialized inside `metadata.pkl` and loaded by the Streamlit app for every real-time prediction.

---

## 🔎 SHAP Explainability

`shap.TreeExplainer` is applied to the XGBoost model on a 2,000-sample stratified subset (500 fraud + 1,500 legitimate). SHAP values explain each individual prediction in terms of feature contributions.

**Global Top Fraud Predictors (from SHAP Summary Plot):**
1. `amt_log` — Large transaction amounts are the strongest single fraud signal
2. `distance_km` — Geographic anomaly between cardholder home and merchant
3. `hour` — Late-night / early-morning transactions (12 AM – 6 AM)
4. `merchant_freq` — Transactions at rarely-visited merchants
5. `category_shopping_net` — Online shopping category

**Example SHAP Waterfall — Fraud Transaction:**
```
Base value (average model output): 0.042

amt_log           +0.41  ████████████  ← Very high amount
distance_km       +0.24  ███████       ← Far from home
hour              +0.19  █████         ← 2:00 AM
merchant_freq     +0.09  ██            ← Rare merchant
category_shop_net +0.06  █             ← Online shopping
age               -0.04                ← Reduces risk slightly

Model output: 0.91 → FRAUD BLOCKED ✓
```

The pre-computed SHAP values are saved in `Models/shap_data.pkl` (642 KB) and loaded by the Streamlit dashboard for instant rendering.

---

## 🌐 Streamlit App — FraudGuard

The champion ensemble model is deployed as the **FraudGuard** interactive dashboard built with Streamlit + Plotly.

**Four tabs:**

| Tab | Content |
|-----|---------|
| 🔍 Transaction Checker | Input form → fraud probability score → risk badge (LOW/MEDIUM/HIGH) → live SHAP waterfall |
| 📊 Model Comparison | AUPRC leaderboard, Precision-Recall curves for all models, SHAP global importance bar |
| 🔗 Fraud Patterns | Interactive FP-Growth rule explorer with lift/confidence/support filters and bubble chart |
| 📈 Dataset EDA | Class distribution, fraud rate by hour, fraud rate by merchant category, feature engineering docs |

All inputs are automatically transformed through the full preprocessing pipeline (log1p, Haversine encoding lookup, frequency medians, StandardScaler) before the model scores them.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.10+
- GPU optional (used CUDA T4 in Colab during training)

### Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detector.git
cd credit-card-fraud-detector
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

> The `Models/` folder contains all pre-trained serialized models. The app loads them on startup and requires no retraining.

### Retrain from Scratch
```bash
# 1. Download the Sparkov dataset from Kaggle
# 2. Set the paths in Notebook.ipynb cell 2:
train_path = 'path/to/fraudTrain.csv'
test_path  = 'path/to/fraudTest.csv'
# 3. Run all cells in order — the last cells save Models/ automatically
jupyter notebook Notebook.ipynb
```

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=2.0.0
lightgbm>=4.3.0
tensorflow>=2.13.0
shap>=0.44.0
mlxtend>=0.23.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.13.0
joblib>=1.3.0
```

---

## 📈 Results Summary

```
Dataset           : Sparkov Synthetic — 1,852,394 transactions
Raw Features      : 23 columns
Engineered Feat.  : 24 features after pipeline
Train / Test      : 1,481,915 / 370,479 (80/20 stratified)
Class Imbalance   : ~191:1 (Legitimate vs Fraud)
SMOTE             : 7,721 → 294,838 fraud samples (training only)
─────────────────────────────────────────────────────────────
Champion Model    : Weighted Ensemble (XGBoost×3 + LightGBM×1)
Ensemble AUPRC    : 0.9517
Ensemble F1       : 0.90  (Precision 0.93, Recall 0.87)
─────────────────────────────────────────────────────────────
XGBoost Alone     : AUPRC 0.9506, Precision 0.94, Recall 0.87
Decision Threshold: 0.5819 (F1-optimized via PR curve)
─────────────────────────────────────────────────────────────
Top FP-Growth Rule: Male + Amt_Very_High + shopping_net
                    + Evening → Fraud (Confidence 97%, Lift 10.23×)
Explainability    : SHAP TreeExplainer — waterfall plots per prediction
Deployment        : Streamlit FraudGuard dashboard
```

---
