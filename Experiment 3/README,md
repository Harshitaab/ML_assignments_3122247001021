# Experiment 3: Regression Analysis using Linear and Regularized Regression Models

**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Degree & Branch:** M.Tech (Integrated) CSE, Semester V  
**Subject:** Machine Learning Algorithms Laboratory (ICS1512)  
**Author:** Harshitaa B (Reg: 3122247001021)  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Date:** 16th August 2026  
**GitHub Repository:** [HarshitaaB-2606](https://github.com/HarshitaaB-2606)

---

## 🎯 Objective
The primary objective of this experiment is to predict a continuous target variable (sanctioned loan amount) using multiple regression algorithms. Specific goals include:
* Implementing Linear Regression as a baseline model.
* Applying Ridge (L2), Lasso (L1), and Elastic Net (L1 + L2) regularized regression techniques.
* Building a robust preprocessing pipeline to handle missing values, scale numerical features, and encode categorical variables.
* Tuning hyperparameters using 5-Fold Cross-Validation and `GridSearchCV`.
* Analyzing the effects of regularization on model complexity, generalization, bias-variance trade-off, and feature sparsity.

## 📊 Dataset Description
The experiment uses a **Loan-Sanction Regression Dataset**.
* **Total Samples:** 592 (Train) / 362 (Test)
* **Target Variable:** `LoanAmount` (Continuous)
* **Numerical Features (4):** ApplicantIncome, CoapplicantIncome, Loan Amount Term, Credit History
* **Categorical Features (7):** Loan ID, Gender, Married, Dependents, Education, Self Employed, Property Area
* **Transformed Features:** 611 (Expanded due to one-hot encoding of high-cardinality categorical variables like `Loan ID`).

## 🛠️ Data Preprocessing & EDA
A unified Scikit-Learn `ColumnTransformer` pipeline was implemented:
* **Numerical Data:** Handled missing values using `median` imputation, followed by `StandardScaler` (critical for regularization penalties).
* **Categorical Data:** Handled missing values using `most_frequent` imputation, followed by `OneHotEncoder`.
* **EDA Highlights:** Identified data distributions, checked for skewness, and analyzed feature-target relationships using scatter plots. High dimensionality from OHE required strict regularization.

## 🧠 Machine Learning Algorithms

### 1. Linear Regression (Baseline)
Models the target as a linear combination of input features using Ordinary Least Squares (OLS). Prone to overfitting on highly dimensional data (611 features).

### 2. Ridge Regression (L2 Penalty)
Adds a squared magnitude penalty to the loss function to shrink coefficients evenly, reducing model variance without dropping features.

### 3. Lasso Regression (L1 Penalty)
Adds an absolute magnitude penalty, inherently performing feature selection by forcing less important coefficients exactly to zero.

### 4. Elastic Net (L1 + L2 Penalty)
Combines Ridge and Lasso penalties, controlled by an overall regularization strength (`alpha`) and a mixing parameter (`l1_ratio`).

---

## 🚀 Key Results & Performance Metrics

### Final Test-Set Evaluation
| Model | MAE | MSE | RMSE | R² Score | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 36.0467 | 2728.8873 | 52.2388 | 0.2734 | **0.2250** |
| **Ridge Regression** | 35.7577 | 2606.4552 | 51.0535 | 0.3060 | 5.4762 |
| **Lasso Regression** | **35.4400** | 2660.1341 | 51.5765 | 0.2917 | 8.6063 |
| **Elastic Net** | 35.7653 | **2587.4734** | **50.8672** | **0.3110** | 2.5264 |

* **Observation:** **Elastic Net** emerged as the best overall model, achieving the highest R² score and the lowest mean squared/root mean squared errors.

---

## ⚙️ Hyperparameter Tuning & Cross-Validation

Hyperparameters were optimized using `GridSearchCV` with 5-Fold Cross-Validation on the training set:

| Model | Search Space | Best Parameters | Best CV R² |
| :--- | :--- | :--- | :--- |
| **Ridge** | `alpha`: [0.01, 0.1, 1, 10, 100] | `alpha = 100` | **0.2964** |
| **Lasso** | `alpha`: [0.001, 0.01, 0.1, 1, 10] | `alpha = 1` | 0.2693 |
| **Elastic Net** | `alpha`: [0.01, 0.1, 1, 10] <br> `l1_ratio`: [0.2, 0.5, 0.8] | `alpha = 1`, `l1_ratio = 0.8` | 0.2919 |

## 💡 Analytical Insights

* **Feature Sparsity (Lasso Effect):** The best Lasso model (`alpha=1`) successfully reduced **602 out of 611** transformed coefficients exactly to zero. This achieved a massive **98.53% sparsity rate**, effectively performing aggressive feature selection while maintaining competitive MAE and R² scores.
* **Bias-Variance Trade-off:** Linear Regression severely overfit the training data (extremely low training MSE but high test MSE). By contrast, regularized models accepted a slightly higher training error in exchange for a significantly lower test error, successfully improving generalization to unseen data.
* **Dimensionality Curse:** The categorical `Loan_ID` feature drastically inflated the feature space to 611 columns. Regularization (specifically Elastic Net and Ridge) was essential to manage this high dimensionality and prevent extreme variance.

## 🏁 Conclusion
The experiment effectively demonstrated the necessity of regularization when dealing with high-dimensional data. Unregularized Linear Regression failed to generalize well on the test set. **Elastic Net** provided the optimal balance between Ridge's coefficient shrinkage and Lasso's feature selection, resulting in the most robust predictive performance (`R² = 0.3110`).
