
# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer churn is a critical business problem where the goal is to identify customers who are likely to stop using a service. In this project, we built an **end-to-end machine learning pipeline** to predict customer churn using structured tabular data.

The focus of this project is **correct ML workflow and evaluation**, not just model accuracy. We emphasize:
- data leakage prevention,
- fair model comparison,
- business-aligned evaluation metrics,
- and justified model selection.

---

## 🎯 Objective

To build, tune, and evaluate multiple machine learning models that predict customer churn, and to **select the most appropriate model** based on **business-relevant metrics**, primarily **recall**.

---

## 🧠 Problem Framing

- **Type:** Binary Classification  
- **Target:** Churn (Yes / No)  
- **Challenge:** Class imbalance (churners are the minority)  
- **Business Risk:**  
  - False Negatives (missed churners) are **more costly** than False Positives  

➡️ This directly influenced our **metric choice and modeling decisions**.

---

## 🚀 Key Features Implemented

- Train–test split with strict separation of unseen data
- End-to-end preprocessing using:
  - `Pipeline`
  - `ColumnTransformer`
- Handling:
  - Missing values
  - Numerical scaling
  - Categorical encoding (Ordinal + One-Hot)
- Stratified Cross-Validation to preserve class distribution
- Model evaluation using **multiple metrics**
- Hyperparameter tuning using `GridSearchCV`
- Comparison of **three different models**
- Final evaluation on the test set using:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC (using probabilities / decision scores)

---

## 🧰 Tech Stack

**Language**
- Python

**Libraries**
- numpy
- pandas
- scikit-learn
- matplotlib

**Core ML Tools**
- Pipeline
- ColumnTransformer
- StratifiedKFold
- GridSearchCV
- classification_report
- ConfusionMatrixDisplay
- roc_auc_score
- RocCurveDisplay

---

## 🧱 Project Architecture (High-Level)

```
Raw Data
   ↓
Train / Test Split
   ↓
Pipeline
   ├── ColumnTransformer
   │     ├── Numerical Pipeline (Imputer + Scaler)
   │     ├── Binary Pipeline (Imputer + OrdinalEncoder)
   │     └── Categorical Pipeline (Imputer + OneHotEncoder)
   ↓
Classifier
   ↓
Cross-Validation & GridSearchCV
   ↓
Final Model Evaluation on Test Set
```

---

## 🤖 Models Trained & Tuned

### 1️⃣ Logistic Regression
- Regularization strength (`C`)
- L2 penalty
- Solver: `lbfgs`

### 2️⃣ Decision Tree Classifier
- Maximum depth
- Minimum samples per split
- Minimum samples per leaf
- Split criterion (`gini`, `entropy`)

### 3️⃣ Support Vector Classifier (SVC)
- Kernel (`linear`, `rbf`)
- Regularization (`C`)
- Kernel coefficient (`gamma`)

---

## 📏 Evaluation Strategy

### Why Recall?
- Missing a churner = lost revenue
- False Negatives are more costly than False Positives
- Recall focuses on identifying **as many churners as possible**

### Supporting Metrics
- Precision
- F1-score
- ROC-AUC (threshold-independent ranking quality)

---

## 🏆 Final Model Selection

**Logistic Regression** was selected as the final model because:
- Highest recall on test data
- Highest ROC-AUC
- Stable and robust
- Easy to interpret
- Low risk of overfitting

---

## ▶️ How to Run the Project

```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## 📌 Key Takeaways

- Pipelines prevent data leakage
- Cross-validation is for evaluation, not final training
- GridSearchCV performs tuning + refit automatically
- Metric choice must align with business cost
- Simpler models often generalize better

---

## 📈 Possible Extensions

- Threshold tuning
- PR-AUC curve
- Cost-sensitive learning
- Feature importance interpretation

---

## 🧠 Learning Outcome

This project demonstrates a **professional, end-to-end machine learning workflow** suitable for real-world classification problems.
