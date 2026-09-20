# Experiment 2: Email Spam/Ham Classification using Naïve Bayes and KNN

**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Degree & Branch:** M.Tech (Integrated) CSE, Semester V  
**Subject:** Machine Learning Algorithms Laboratory (ICS1512)  
**Author:** Harshitaa B (Reg: 3122247001021)  

---

## 🎯 Objective
The primary objective of this experiment is to classify emails into spam and ham categories using Naïve Bayes and K-Nearest Neighbour (KNN) classifiers. Specific goals include:
* Comparing Gaussian, Multinomial, and Bernoulli Naïve Bayes models.
* Studying the effect of varying `k` in KNN.
* Comparing KDTree and BallTree neighbor-search algorithms.
* Optimizing KNN hyperparameters using `GridSearchCV` and `RandomizedSearchCV`.
* Evaluating computational complexity, execution time, and K-Fold Cross Validation.

## 📊 Dataset Description
The experiment uses the **Spambase Dataset**.
* **Number of Samples:** 4601
* **Number of Features:** 57 (Numerical)
* **Classes:** 2 (Class 0: Ham, Class 1: Spam)
* **Train-Test Split:** 80:20 (Stratified)

## 🛠️ Data Preprocessing & EDA
* **Missing Values:** Checked and handled prior to model training.
* **Feature Scaling:** Standardization (Z-score scaling) was applied since KNN relies heavily on distance metrics (e.g., Euclidean, Manhattan).
* **EDA Highlights:** Analyzed class distribution, feature distributions, and the correlation matrix to understand feature relationships.

## 🧠 Machine Learning Algorithms

### 1. Naïve Bayes
A probabilistic classification algorithm based on Bayes’ theorem with conditional independence assumptions:
`P(C|X) = [P(X|C) * P(C)] / P(X)`
* **Variants Tested:** Gaussian, Multinomial, and Bernoulli Naïve Bayes.

### 2. K-Nearest Neighbour (KNN)
A lazy learning algorithm that classifies a sample based on the majority class among its `k` nearest neighbors using Minkowski distance (Euclidean for `p=2`, Manhattan for `p=1`).
* **k-values Tested:** 1, 3, 5, 7, 9, 11.

---

## 🚀 Key Results & Performance Metrics

### Naïve Bayes Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gaussian NB** | 0.8328 | 0.7146 | **0.9587** | 0.8188 | 0.9376 |
| **Multinomial NB** | **0.8958** | **0.9349** | 0.7906 | **0.8567** | **0.9612** |
| **Bernoulli NB** | 0.8795 | 0.8684 | 0.8182 | 0.8426 | 0.9457 |

* **Observation:** Multinomial NB yielded the best overall accuracy and precision among NB variants, while Gaussian NB had the highest recall.

### KNN Performance (Varying k)
| k | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.8792 | 0.8623 | 0.8707 | 0.8665 | 0.8926 |
| 5 | 0.9077 | 0.8861 | **0.8788** | 0.8824 | 0.9506 |
| **11** | **0.9099** | **0.9000** | 0.8678 | **0.8836** | **0.9622** |

* **Observation:** Performance stabilized at higher `k` values, with `k=11` emerging as the optimal configuration for accuracy, F1-score, and ROC-AUC. 

### KDTree vs. BallTree (KNN k=11)
| Algorithm | Accuracy | Training Time (s) | Prediction Time (s) |
| :--- | :--- | :--- | :--- |
| **KDTree** | 0.9099 | 0.0386 | 0.3290 |
| **BallTree** | 0.9099 | **0.0268** | **0.3192** |

* **Observation:** Both algorithms yielded identical accuracy, but BallTree was slightly faster in both training and prediction for this dataset.

---

## ⚙️ Hyperparameter Tuning

| Search Method | Best CV Accuracy | Execution Time (s) | Best Parameters |
| :--- | :--- | :--- | :--- |
| **GridSearchCV** | 0.9000 | 134.82 | `k=7`, Manhattan (`p=1`), `weights='distance'` |
| **RandomizedSearchCV** | **0.9004** | **26.16** | `k=18`, Euclidean (`p=2`), `weights='distance'`, KDTree |

* **Observation:** `RandomizedSearchCV` found a slightly better configuration in a fraction of the time compared to `GridSearchCV`.

## ⏱️ Computational Complexity Analysis
* **KNN Time Profile:** KNN required the least training time (~0.010s) but had the highest prediction time (~0.220s). This aligns with theoretical complexity, as KNN is a lazy learner (O(1) training, O(nd) prediction).
* **Naïve Bayes Time Profile:** Fast in both training and prediction, scaling linearly with dataset size.

## 🏁 Conclusion
The experiment successfully demonstrated the use of probabilistic and distance-based classifiers for spam detection. **KNN (k=11)** proved to be the most accurate model (90.99%), outperforming all Naïve Bayes variants. However, Naïve Bayes algorithms (especially Multinomial) offer highly competitive accuracy with significantly lower prediction times, making them highly suitable for time-sensitive, large-scale text classification tasks.
