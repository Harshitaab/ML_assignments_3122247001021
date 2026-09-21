# Experiment 4: Binary Classification using Linear and Kernel-Based Models

**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Degree & Branch:** M.Tech (Integrated) CSE, Semester V  
**Subject:** Machine Learning Algorithms Laboratory (ICS1512)  
**Author:** Harshitaa B (Reg: 3122247001021)  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Date:** 16th August 2026  
**GitHub Repository:** [HarshitaaB-2606](https://github.com/HarshitaaB-2606) *(Placeholder Link)*

---

## 🎯 Objective
The primary objective of this experiment is to develop and evaluate binary classification models to classify emails as spam or ham. Specific goals include:
* Implementing a linear probabilistic classifier (**Logistic Regression**) and a margin-based classifier (**Support Vector Machine**).
* Evaluating the impact of different SVM kernels (Linear, Polynomial, RBF, Sigmoid).
* Applying hyperparameter tuning using cross-validation to optimize model configurations.
* Analyzing the models using accuracy, precision, recall, and F1-score.
* Comparing Cross-Validation (CV) performance versus final Test-Set performance to understand model generalization.

## 📊 Dataset Description
The experiment uses the **Spambase Dataset**.
* **Task:** Binary Classification (Spam vs. Ham)
* **Features:** Numerical features extracted from email content (e.g., word frequencies, character frequencies).
* **Target Variable:** 0 (Ham) / 1 (Spam)

## 🛠️ Data Preprocessing & EDA
* **Feature Scaling:** Since both Logistic Regression and SVM are distance/magnitude-sensitive, standardizing the numerical features (e.g., using `StandardScaler`) was a critical preprocessing step.
* **EDA Highlights:** Analyzed the class distribution (to check for imbalances) and generated a correlation matrix to understand the relationships between different email features prior to training.

## 🧠 Machine Learning Algorithms

### 1. Logistic Regression
A probabilistic classifier that uses the sigmoid function to model the probability of the positive class.
* **Hyperparameters Tuned:** Regularization penalty (L1 vs. L2), inverse regularization strength (`C`), and solvers (`liblinear`, `saga`).

### 2. Support Vector Machine (SVM)
A margin-based classifier that finds the optimal hyperplane to separate classes while maximizing the margin.
* **Kernels Evaluated:** Linear, Polynomial, Radial Basis Function (RBF), and Sigmoid.
* **Hyperparameters Tuned:** Penalty parameter (`C`), Kernel type, and Kernel coefficient (`gamma`).

---

## 🚀 Key Results & Performance Metrics

### 1. SVM Kernel Comparison (Untuned)
| Kernel | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear** | **0.9294** | 0.9209 | **0.8981** | **0.9093** | 0.5773 |
| **RBF** | 0.9273 | 0.9277 | 0.8843 | 0.9055 | **0.2895** |
| **Sigmoid** | 0.8849 | 0.8599 | 0.8457 | 0.8528 | 0.3540 |
| **Polynomial** | 0.7796 | **0.9598** | 0.4601 | 0.6220 | 0.5557 |

* **Observation:** The Linear kernel provided the best baseline performance. The Polynomial kernel achieved high precision but suffered from a severely low recall, missing a significant number of positive samples.

### 2. Final Tuned Models Comparison (Test Set)
| Model | Best Hyperparameters | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C=1`, Penalty=`L2`, Solver=`saga` | **0.9305** | **0.9211** | **0.9008** | **0.9109** |
| **SVM** | `C=10`, Kernel=`RBF`, `gamma='scale'` | 0.9207 | 0.9143 | 0.8815 | 0.8976 |

### 3. Five-Fold Cross-Validation vs. Test Performance
| Model | Average CV Accuracy | Final Test Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.9244 | **0.9305** |
| **SVM** | **0.9337** | 0.9207 |

---

## 💡 Analytical Insights

* **The CV vs. Test Paradox:** SVM achieved a higher average accuracy during 5-fold cross-validation (93.37%) compared to Logistic Regression (92.44%). However, when evaluated on the unseen test set, Logistic Regression generalized better (93.05% vs 92.07%). This highlights that validation scores do not always perfectly correlate with final test-set rankings.
* **Hyperparameter Tuning Impact:** Tuning yielded a small but consistent improvement across all metrics for Logistic Regression (Accuracy improved from 92.94% to 93.05%). 
* **Computational Efficiency:** Logistic Regression is highly efficient for this dataset. Its baseline model trained in just **0.1428 seconds**. However, tuning LR took longer (~289s) than tuning SVM (~120s) due to the specific solver and parameter combinations tested.

## 🏁 Conclusion
The experiment successfully demonstrated the application of linear and kernel-based models for spam classification. While SVM with an RBF kernel showed the strongest validation performance during hyperparameter tuning, **Logistic Regression (with L2 regularization)** ultimately proved to be the best-performing model on the final test set, achieving the highest accuracy (93.05%) and F1-score (91.09%).
