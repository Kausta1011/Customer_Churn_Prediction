
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

## 📊 Results & Interpretation

### Cross-Validation Performance (Recall)

All models were tuned using **GridSearchCV with stratified cross-validation**, optimizing for **recall** due to the high cost of missing churners.

| Model | Best CV Recall |
|------|---------------|
| Logistic Regression | **0.548** |
| Support Vector Classifier (Linear) | 0.531 |
| Decision Tree | 0.524 |

**Interpretation:**
- All three models achieved **similar recall**, indicating that model choice has less impact than feature quality.
- Logistic Regression achieved the **highest recall**, although the margin was small.
- This suggests the problem is **feature-limited rather than model-limited**.

---

### Test-Set ROC-AUC Performance

ROC-AUC was computed using **predicted probabilities or decision scores** on the unseen test set.

| Model | Test ROC-AUC |
|-----|-------------|
| Logistic Regression | **0.84** |
| Support Vector Classifier | 0.83 |
| Decision Tree | 0.76   |

**Interpretation:**
- Logistic Regression showed the **best class-separation ability**, meaning it ranks churners higher than non-churners more consistently.
- Decision Trees exhibited the weakest generalization performance, likely due to overfitting.
- SVC performed competitively but did not surpass Logistic Regression.

---

### Confusion Matrix Analysis

Confusion matrices were analyzed to understand **False Negatives (FN)** and **False Positives (FP)**.

**Key Observations:**
- Logistic Regression produced **fewer False Negatives**, which is critical for churn prediction.
- While False Positives still exist, this trade-off is acceptable because contacting a non-churner is less costly than losing a customer.

---

### Final Model Selection

Despite similar performance across models, **Logistic Regression** was selected as the final model due to:

- Highest recall on cross-validation and test data
- Best ROC-AUC score
- Greater stability and lower overfitting risk
- Strong interpretability for business stakeholders
- Simpler deployment and maintenance

> When performance is comparable, **robustness and interpretability outweigh model complexity**.

---

### Business Implications

- The selected model prioritizes **identifying as many churners as possible**, reducing customer loss.
- The model can support customer retention strategies by flagging high-risk users early.
- Future improvements are more likely to come from **feature engineering or decision-threshold tuning** rather than more complex models.

---

### 🧠 Key Takeaway

This project demonstrates that **metric selection, proper evaluation strategy, and interpretability** are more important than model complexity in real-world machine learning systems.



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
