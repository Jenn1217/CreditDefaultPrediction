# Credit Default Prediction with LightGBM

A machine learning project for **credit default risk prediction** based on large-scale borrower behavior data.
The project implements a complete modeling pipeline including **data preprocessing, feature engineering, model comparison, and interpretability analysis using SHAP**.

[English](README.md) | [中文](README_CN.md)

## Project Overview

Credit risk assessment is a critical task in financial services.
This project builds a **credit default prediction model** using borrower behavior data and compares several machine learning algorithms.

The main objective is to:

* Predict whether a borrower will **default on a loan**
* Compare the performance of different ML models
* Identify the **key factors affecting default risk**

The final **LightGBM model achieved an AUC of 0.86**, outperforming baseline models.

---

# Dataset

The dataset contains **large-scale borrower behavior data**, including:

* Personal attributes
* Loan information
* Credit history
* Financial indicators

Key variables include:

* loan amount
* interest rate
* credit history
* repayment status
* issue date
* earliest credit line

Target variable:

```
isDefault
```

* 0 → Non-default
* 1 → Default

---

# Project Structure

```
credit-default-prediction-lightgbm
│
├── finalresult.ipynb        # Main notebook (data processing + modeling)
├── train.csv                # Training dataset
├── testA.csv                # Test dataset
└── README.md                # Project documentation
```

---

# Workflow

The project follows a standard **machine learning modeling pipeline**.

### 1 Data Preprocessing

* Load training and test datasets
* Handle missing values
* Convert date features to datetime
* Extract useful time features

Example:

* `issueDate` → time difference features
* `earliesCreditLine` → credit history length

---

### 2 Feature Engineering

Key feature transformations include:

* Date feature transformation
* Credit history extraction
* Categorical variable encoding
* Numerical feature selection

Libraries used:

```
Pandas
NumPy
Scikit-learn
```

---

### 3 Model Training

Multiple machine learning models were trained and compared:

| Model               | Description                           |
| ------------------- | ------------------------------------- |
| Logistic Regression | Linear baseline model                 |
| Random Forest       | Ensemble tree model                   |
| XGBoost             | Gradient boosting model               |
| LightGBM            | Efficient gradient boosting framework |

The models were evaluated using:

```
AUC (Area Under ROC Curve)
```

---

### 4 Model Performance

| Model               | AUC             |
| ------------------- | --------------- |
| Logistic Regression | baseline        |
| Random Forest       | improved        |
| XGBoost             | strong          |
| **LightGBM**        | **0.86 (best)** |

LightGBM achieved the **best performance** in predicting credit default risk.

---

# Model Interpretability

To improve model transparency, the project uses:

```
SHAP (SHapley Additive exPlanations)
```

SHAP helps identify:

* Key features influencing default probability
* Positive and negative risk factors
* Feature importance ranking

This provides **explainability for financial risk models**, which is important for real-world credit systems.

---

# Tech Stack

Python libraries used:

```
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
LightGBM
SHAP
```

---

# Results

Key findings:

* Gradient boosting models outperform linear models
* Borrower credit history is a strong predictor
* Feature engineering significantly improves model performance
* LightGBM provides the best balance between performance and efficiency

---

# Future Work

Potential improvements include:

* Hyperparameter tuning
* Cross-validation
* Feature selection optimization
* Deep learning models
* More advanced explainability analysis

---

# Author

Katrina
Graduate Student | AI / Financial Technology / Machine Learning
