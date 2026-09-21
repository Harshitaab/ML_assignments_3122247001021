# Experiment 6 – Bagging, Boosting, and Stacked Ensemble Models

## 📌 Experiment Overview

This experiment studies and implements ensemble learning techniques for binary classification using the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**.

The following ensemble methods are implemented and compared:

* Decision Tree
* Bagging
* AdaBoost
* Gradient Boosting
* Stacked Ensemble

The models are evaluated using **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**. Hyperparameters are selected using **Stratified 5-Fold Cross-Validation**.

---

## 🎯 Objectives

* Understand Bagging, Boosting, and Stacking ensemble techniques.
* Implement Bagging and Boosting classifiers.
* Construct a heterogeneous Stacked Ensemble.
* Perform hyperparameter tuning using 5-fold cross-validation.
* Compare the models using multiple classification metrics.
* Analyze the effect of ensemble methods on bias, variance, stability, and generalization.

---

## 📊 Dataset

### Wisconsin Diagnostic Breast Cancer (WDBC)

| Property         | Description                        |
| ---------------- | ---------------------------------- |
| Dataset          | Wisconsin Diagnostic Breast Cancer |
| Samples          | 569                                |
| Features         | 30 numerical features              |
| Classes          | Malignant, Benign                  |
| Missing Values   | None                               |
| Train-Test Split | 80:20                              |
| Split Type       | Stratified                         |
| Random State     | 42                                 |

The dataset contains numerical measurements describing characteristics of cell nuclei.

---

## 🤖 Algorithms

### 1. Decision Tree

A Decision Tree is used as the baseline classification model.

### 2. Bagging

Bagging (Bootstrap Aggregation) trains multiple base estimators using bootstrap samples and combines their predictions. It primarily helps reduce model variance.

### 3. AdaBoost

AdaBoost builds classifiers sequentially, giving greater emphasis to observations that were incorrectly classified by previous learners.

### 4. Gradient Boosting

Gradient Boosting sequentially builds learners to reduce residual prediction errors and improve classification performance.

### 5. Stacked Ensemble

The Stacked Ensemble combines predictions from heterogeneous base learners:

* SVM
* Gaussian Naive Bayes
* Decision Tree

A **Logistic Regression** model is used as the meta-learner.

---

## ⚙️ Preprocessing

The following preprocessing steps are performed:

* Check for missing values.
* Encode the binary target.
* Apply `StandardScaler` for the SVM pipeline.
* Use tree-based models without feature scaling.
* Perform an 80:20 stratified train-test split.
* Set `random_state = 42`.
* No additional feature engineering is performed.

---

## 🔧 Hyperparameter Tuning

Hyperparameters are selected using **Stratified 5-Fold Cross-Validation** with mean F1 Score as the primary selection metric.

### Bagging

* Number of estimators: `3, 10, 25, 50`
* Maximum samples: `0.6, 0.8, 1.0`
* Maximum features: `0.7, 1.0`

### AdaBoost

* Number of estimators: `25, 50, 100, 200`
* Learning rate: `0.01, 0.1, 0.5, 1.0`

### Gradient Boosting

* Number of estimators: `50, 100, 150`
* Learning rate: `0.03, 0.1, 0.2`
* Maximum depth: `1, 2, 3`

---

## 📈 Evaluation Metrics

The models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **ROC-AUC**
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## 📊 Final Results

| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ----------------- | -------: | --------: | -----: | -------: | ------: |
| Decision Tree     |   0.9123 |    0.9559 | 0.9028 |   0.9286 |  0.9157 |
| Bagging           |   0.9561 |    0.9589 | 0.9722 |   0.9655 |  0.9927 |
| AdaBoost          |   0.9561 |    0.9467 | 0.9861 |   0.9660 |  0.9818 |
| Gradient Boosting |   0.9649 |    0.9595 | 0.9861 |   0.9726 |  0.9947 |
| Stacked Ensemble  |   0.9737 |    0.9726 | 0.9861 |   0.9793 |  0.9937 |

The values above are the final test-set results reported in the experiment.

---

## 🔍 Observations

* Ensemble models achieved higher test performance than the Decision Tree baseline.
* Bagging reduces variance by aggregating predictions from multiple bootstrap-trained estimators.
* Boosting methods sequentially focus on correcting prediction errors.
* Gradient Boosting achieved a test accuracy of **96.49%** and F1 Score of **97.26%**.
* The Stacked Ensemble achieved a test accuracy of **97.37%** and F1 Score of **97.93%**.
* The ensemble models achieved high ROC-AUC values, indicating strong class discrimination on the test set.
* Stacking combines different modelling approaches through a meta-learner and can exploit complementary predictions.

---

## 🏆 Final Result

The experiment successfully demonstrates three ensemble learning strategies:

1. **Bagging** – variance reduction through bootstrap aggregation.
2. **Boosting** – sequential error correction.
3. **Stacking** – combining heterogeneous learners through a meta-learner.

Based on the reported test-set metrics, the **Stacked Ensemble recorded the highest Accuracy (97.37%) and F1 Score (97.93%)** among the models evaluated in this experiment.

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook / JupyterLab
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## 📁 Suggested Project Structure

```text
Experiment-6/
│
├── Experiment_6.ipynb
├── README.md
└── report/
    └── experiment_6_report.pdf
```

---

## 📚 References

* Scikit-learn – Ensemble Methods
* Scikit-learn – BaggingClassifier
* Scikit-learn – AdaBoostClassifier
* Scikit-learn – GradientBoostingClassifier
* Scikit-learn – StackingClassifier
* UCI Machine Learning Repository – Breast Cancer Wisconsin Diagnostic Dataset
