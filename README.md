# 🫀 Why Linear Models Fail?
### A Study of Non-Linearity in Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.2-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project investigates **why linear models fail on real-world medical data**
by comparing four machine learning algorithms on the
[UCI Heart Disease (Cleveland) Dataset](https://www.kaggle.com/datasets/hamnawaseem112222222/uci-heart-disease-dataset).

The core question:
> *"Can a straight line separate patients with heart disease from those without —
> and if not, what can?"*

---

## 🎯 Objectives

- Demonstrate the **failure cases of linear models** on non-linear medical data
- Implement and compare **4 machine learning algorithms**
- Visualize **decision boundaries** to build intuition
- Explain **why non-linear models improve performance**
- Analyze **feature importance** for medical interpretability

---

## 🧠 Algorithms Compared

| Model | Type | Key Idea |
|-------|------|----------|
| **Logistic Regression** | Linear | Weighted sum → sigmoid → probability |
| **k-Nearest Neighbors** | Non-linear | "Patients similar to you had this outcome" |
| **Support Vector Machine** | Linear (max-margin) | Find the boundary with the widest safety gap |
| **XGBoost** | Non-linear ensemble | 91 sequential trees, each correcting previous errors |

---

## 📊 Results

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| Logistic Regression | 84.0% | 86.7% |
| k-NN (k=10) | 89.9% | 88.3% |
| SVM (C=1.0) | 83.5% | **90.0%** ✅ |
| XGBoost (91 trees) | 88.2% | 83.3% |

**Key finding:** SVM achieved the highest test accuracy (90%) on this small dataset
(297 examples), demonstrating that geometric constraints (maximum margin) can
outperform complex ensemble methods when data is limited.

---

## 🗂️ Project Structure
heart-disease-nonlinearity/

│

├── data/

│   └── heart_disease_cleveland.csv    # UCI Heart Disease Dataset

│

├── src/

│   ├── data_loader.py                 # Load, clean, standardize data

│   ├── logistic_regression.py         # LR from scratch (NumPy)

│   ├── knn.py                         # k-NN from scratch (NumPy)

│   ├── svm.py                         # SVM from scratch (NumPy)

│   ├── visualize.py                   # Decision boundary plots

│   └── main.py                        # Runs all models + comparisons

│

├── results/                           # Generated plots (auto-created)

│   ├── lr_age_thalach.png

│   ├── knn_age_thalach.png

│   ├── svm_age_thalach.png

│   ├── xgboost_age_thalach.png

│   ├── xgb_feature_importance.png

│   └── xgb_learning_curve.png

│

├── .gitignore

└── README.md

---

## 🔬 Dataset

**UCI Heart Disease (Cleveland)**

| Property | Value |
|----------|-------|
| Samples | 297 (after cleaning) |
| Features | 13 |
| Target | Binary (0 = no disease, 1 = disease) |
| Class balance | 160 / 137 |
| Missing values | 6 rows dropped (ca, thal) |

**Key features:**

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `sex` | Sex (1=male, 0=female) | Binary |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `thalach` | Maximum heart rate achieved | Continuous |
| `ca` | Number of blocked vessels (0-4) | Ordinal |
| `thal` | Thalassemia type | Categorical |
| `target` | Heart disease present | **Target** |

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/BahaKh-IO/heart-disease-nonlinearity.git
cd heart-disease-nonlinearity
```

### 2. Install dependencies

```bash
pip install numpy matplotlib scikit-learn pandas xgboost
```

### 3. Add the dataset

Download `heart_disease_cleveland.csv` from
[Kaggle](https://www.kaggle.com/datasets/hamnawaseem112222222/uci-heart-disease-dataset)
and place it in the `data/` folder.

### 4. Run the project

```bash
python src/main.py
```

---

## 📈 Visualizations

### Decision Boundaries (Age vs Max Heart Rate)

The project generates side-by-side decision boundary plots for all 4 models,
showing how each algorithm carves up the feature space differently:

- **Logistic Regression** → one straight line
- **k-NN** → curved, irregular regions with local islands
- **SVM** → straight line, but optimally positioned (maximum margin)
- **XGBoost** → stepped, complex non-linear boundary

### XGBoost Learning Curve

Shows log-loss decreasing over 91 boosting rounds on both
train and test sets — visual proof that sequential error
correction works.

### XGBoost Feature Importance

Top features identified by XGBoost:

| Rank | Feature | Importance | Medical Meaning |
|------|---------|------------|-----------------|
| 1 | `ca` | 0.194 | Blocked vessels — direct structural evidence |
| 2 | `thal` | 0.193 | Thalassemia type — blood disorder indicator |
| 3 | `cp` | ~0.15 | Chest pain type — primary symptom |

---

## 💡 Key Insights

### Why Logistic Regression Fails Here
Heart disease risk is NON-LINEAR:

→ thalach (max heart rate) has a U-shaped

relationship with risk

→ age interacts with cholesterol

(interaction effects)

→ blocked vessels (ca) have threshold effects
A single straight line cannot represent

these complex, multi-directional patterns.

### What SVM Fixes
SVM doesn't just find A boundary —

it finds THE boundary with maximum margin.
This geometric constraint acts as natural

regularization, making SVM robust to

the small dataset size (297 examples).
Result: 90% test accuracy vs LR's 86.7%

### Why XGBoost Underperformed Here
XGBoost's strength = large, complex datasets

This dataset = only 297 examples
Even with regularization (λ=1.0, α=0.05)

and early stopping (round 91), the gap

between train (88.2%) and test (83.3%)

shows that 91 sequential trees introduce

more variance than this dataset can support.
Lesson: "More complex" ≠ "better"

Match model complexity to dataset size.

---

## 🔧 Implementation Notes

| Component | Implementation |
|-----------|---------------|
| Logistic Regression | From scratch — NumPy only |
| k-NN | From scratch — NumPy only |
| SVM | From scratch — NumPy only (hinge loss + gradient descent) |
| XGBoost | `xgboost` library (approved by instructor) |
| Data loading/splitting | `scikit-learn` (dataset + train_test_split only) |
| Visualization | `matplotlib` only |

---

## 📚 Concepts Demonstrated

- ✅ Linear vs non-linear decision boundaries
- ✅ Bias-variance tradeoff (visualized with accuracy gaps)
- ✅ Margin maximization (SVM)
- ✅ Sequential error correction (boosting)
- ✅ Feature importance analysis
- ✅ Data preprocessing (standardization, missing value handling)
- ✅ Overfitting detection and regularization
- ✅ Learning curves

---

## 👤 Author

**Baha Eddine Khantouch**
Computer Science Engineering Student
Research Lab Project — Machine Learning

---

## 📄 License

This project is for educational purposes.
Dataset credits: UCI Machine Learning Repository
(Cleveland Heart Disease Dataset)
