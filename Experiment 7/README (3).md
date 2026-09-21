# Experiment 6: Dimensionality Reduction and Model Evaluation Using PCA

**Institution:** Sri Sivasubramaniya Nadar College of Engineering (Assuming based on previous format)
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Harshitaa B (Reg: 3122247001021)  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Submission Date:** 31st August 2026  
**GitHub Repository:** [ML_assignments_3122247001021](https://github.com/Harshitaab/ML_assignments_3122247001021)

---

## 1. Objective
The objective of this experiment is to study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of different machine-learning classifiers. Two experimental settings are compared: 
1. Models trained using the original standardized feature space without PCA.
2. Models trained using a PCA-reduced feature space. 

Hyperparameter tuning using `GridSearchCV` and stratified five-fold cross-validation are employed to make the comparison systematic. The experiment evaluates SVM, Naïve Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and a Stacking ensemble.

## 2. Problem Statement
The problem is to determine whether reducing the dimensionality of the feature space improves or degrades classification performance and validation stability. The input consists of numerical measurements from the Breast Cancer Wisconsin Diagnostic dataset, while the output is the predicted binary class. The experiment first establishes a baseline using the original 30 features and then transforms the standardized training data into principal components. The reduced representation is selected to retain approximately 95% of the training variance. The resulting classifiers are compared using test-set accuracy, precision, recall, F1-score, five-fold cross-validation mean F1-score, and cross-validation standard deviation.

## 3. Dataset Description
The Breast Cancer Wisconsin Diagnostic dataset supplied by `scikit-learn` was used. It contains numerical measurements derived from digitized images of breast-mass cell nuclei. The target contains two classes, malignant and benign. The dataset contains 569 observations and 30 numerical features, with 212 malignant and 357 benign observations. No missing values were found in the predictor matrix. The data were divided using an 80:20 stratified train-test split, resulting in 455 training samples and 114 test samples.

| Dataset Property | Value |
| :--- | :--- |
| **Dataset Name** | Breast Cancer Wisconsin Diagnostic dataset |
| **Dataset Source** | scikit-learn (`load_breast_cancer`) |
| **Number of Samples** | 569 |
| **Number of Features** | 30 numerical features |
| **Number of Classes** | 2 |
| **Classes** | Malignant (212), Benign (357) |
| **Missing Values** | 0 |
| **Train-Test Split** | 80% training (455), 20% testing (114) |
| **Random State** | 42 |

## 4. Exploratory Data Analysis

### 4.1 Dataset Overview and Statistical Summary
The first observations and descriptive statistics were inspected before model training. All predictor variables are numerical. The missing-value check returned zero, so no imputation procedure was required. The descriptive statistics also showed that the numerical features have substantially different scales. This makes standardization particularly important because PCA is variance-based and distance- or margin-based algorithms such as KNN and SVM are sensitive to feature scale.

### 4.2 Class Distribution
> *Figure 1: Distribution of the malignant and benign target classes.*

**Inference:** The dataset contains 212 malignant observations and 357 benign observations. Thus, the classes are not perfectly balanced, although both classes contain a substantial number of examples. Because accuracy alone can hide differences in class-wise performance, precision, recall, and F1-score are also reported. The stratified train-test split and `StratifiedKFold` procedure preserve approximately the same class proportions across evaluation subsets.

### 4.3 Exploratory Analysis Summary
The EDA confirms that the dataset is numerical, contains no missing predictor values, and has features on different scales. These characteristics make the dataset appropriate for demonstrating standardization followed by PCA. A separate correlation-matrix visualization was not generated in the executed notebook; therefore, no additional correlation figure is claimed as an experimental output.

## 5. Data Preprocessing

### 5.1 Train-Test Split
The data were split into 80% training and 20% testing subsets using `train_test_split` with `stratify=y` and `random_state=42`. The training set contains 455 observations and is used for hyperparameter tuning and cross-validation. The 114-sample test set is kept independent for the final evaluation.

### 5.2 Standardization
Each numerical feature was standardized using:
$$ z = \frac{x - \mu}{\sigma} $$
where $x$ is the original feature value, $\mu$ is the feature mean, and $\sigma$ is its standard deviation. Standardization prevents features with larger numerical scales from dominating PCA and also provides a consistent representation for the classifiers.

### 5.3 Pipeline and Data Leakage Prevention
For the **No-PCA** setting, the pipeline is:
`Original Features` $\rightarrow$ `StandardScaler` $\rightarrow$ `Classifier`

For the **PCA** setting, the pipeline is:
`Original Features` $\rightarrow$ `StandardScaler` $\rightarrow$ `PCA` $\rightarrow$ `Classifier`

The scaler and PCA transformation are fitted within the cross-validation pipeline. Consequently, information from a validation fold is not used to fit preprocessing transformations for that fold, reducing the risk of data leakage.

## 6. Principal Component Analysis
PCA transforms the original correlated feature space into a new orthogonal coordinate system called principal components. The first component captures the maximum possible variance; each subsequent component captures the maximum remaining variance subject to orthogonality with the preceding components.

If the standardized feature vector is $x$ and the principal directions are stored in the matrix $W$, the transformed representation can be expressed as:
$$ z = W^T x $$

The explained-variance ratio of component $i$ describes the fraction of total standardized data variance captured by that component.

### 6.1 Component Selection
The original feature space contains 30 features. The cumulative explained-variance analysis selected 10 principal components to retain at least 95% of the training variance. The actual retained cumulative variance was **95.2677%**. Thus, PCA reduced the dimensionality from 30 to 10 features while retaining approximately 95% of the variance.

> *Figure 2: Cumulative explained variance as a function of the number of principal components.*

**Inference:** The cumulative curve increases as additional principal components are included. The horizontal reference corresponds to the 95% variance target, and the selected point occurs at 10 components. Therefore, ten components are sufficient to retain at least the required variance. This provides an objective basis for selecting the PCA dimensionality instead of choosing the number of components arbitrarily.

> *Figure 3: Explained variance ratio of the individual principal components.*

**Inference:** The first few principal components capture substantially more variance than later components. The decreasing contribution of later components indicates redundancy in the original 30-dimensional representation. This demonstrates why a lower-dimensional representation can preserve much of the variation while removing several low-contribution directions.

## 7. Machine Learning Algorithms

### 7.1 Support Vector Machine
SVM seeks a decision boundary that maximizes the margin between classes. For a linear classifier, the decision function can be represented as:
$$ f(x) = w^T x + b $$
The parameter $C$ controls the trade-off between maximizing the margin and penalizing classification errors. 
* **Advantages:** effective in high-dimensional spaces, strong margin-based classification.
* **Limitations:** sensitive to hyperparameter selection and computationally expensive for larger datasets.

### 7.2 Naïve Bayes
Gaussian Naïve Bayes applies Bayes’ theorem while assuming conditional independence among features:
$$ P(c | x) \propto P(c) \prod_i P(x_i | c) $$
* **Advantages:** simple, fast, and effective probabilistic baseline.
* **Limitations:** conditional-independence assumption may not hold for correlated features.

### 7.3 K-Nearest Neighbors
* **Advantages:** simple, non-parametric, and capable of modelling nonlinear boundaries.
* **Limitations:** sensitive to scaling, the choice of $k$, and irrelevant or redundant dimensions.

### 7.4 Logistic Regression
Models the probability of the positive class using the sigmoid function:
$$ P(y = 1 | x) = \frac{1}{1 + \exp[-(w^T x + b)]} $$
* **Advantages:** interpretable, computationally efficient.
* **Limitations:** limited when complex nonlinear relationships are essential.

### 7.5 Decision Tree
* **Advantages:** interpretable and capable of modelling nonlinear relationships without requiring feature scaling.
* **Limitations:** individual trees can be unstable and prone to overfitting.

### 7.6 Random Forest
* **Advantages:** robust, nonlinear, and less prone to overfitting than a single tree.
* **Limitations:** larger computational requirements and reduced interpretability.

### 7.7 AdaBoost
* **Advantages:** can improve weak learners substantially.
* **Limitations:** sensitive to noisy or difficult observations.

### 7.8 Gradient Boosting
* **Advantages:** powerful nonlinear modelling and strong predictive performance.
* **Limitations:** sensitive to hyperparameters and computationally expensive.

### 7.9 XGBoost
* **Advantages:** strong predictive performance, regularization, and efficient tree boosting.
* **Limitations:** hyperparameter tuning can be complex.

### 7.10 Stacking
Combines predictions from multiple base learners (SVM, KNN, Random Forest) and supplies them to a meta-learner (Logistic Regression).

## 8. Experimental Setup

| Parameter | Value |
| :--- | :--- |
| **Python Environment** | Python-based Jupyter Notebook |
| **Dataset Library** | scikit-learn |
| **Numerical Processing** | NumPy, pandas |
| **Visualization** | Matplotlib |
| **Machine Learning** | scikit-learn, XGBoost |
| **Validation** | StratifiedKFold, 5 folds |
| **Hyperparameter Tuning**| GridSearchCV |
| **Scoring for Selection**| F1-score |
| **Train-Test Split** | 80:20, stratified |
| **Random State** | 42 |
| **PCA Target** | At least 95% variance |
| **Selected Components** | 10 |

## 9. Hyperparameter Selection
GridSearchCV was used for the nine individual classifiers using stratified five-fold cross-validation and F1-score as the metric.

**Table 2: Best hyperparameters selected for each model and setting.**

| Model | Setting | Best Configuration | CV F1 | Std. |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | No PCA | `C=0.1, kernel=linear, gamma=scale` | 0.9810 | 0.0082 |
| **SVM** | With PCA | `C=10, kernel=linear, gamma=scale` | 0.9843 | 0.0065 |
| **Naive Bayes** | No PCA | `var_smoothing=10^-9` | 0.9478 | 0.0228 |
| **Naive Bayes** | With PCA | `var_smoothing=10^-9` | 0.9433 | 0.0134 |
| **KNN** | No PCA | `k=3, weights=uniform, metric=manhattan` | 0.9774 | 0.0104 |
| **KNN** | With PCA | `k=7, weights=uniform, metric=manhattan` | 0.9742 | 0.0152 |
| **Logistic Regression** | No PCA | `C=0.1, solver=lbfgs` | 0.9861 | 0.0042 |
| **Logistic Regression** | With PCA| `C=1, solver=lbfgs` | 0.9861 | 0.0042 |
| **Decision Tree** | No PCA | `max_depth=3, min_samples_split=5` | 0.9418 | 0.0034 |
| **Decision Tree** | With PCA| `max_depth=None, min_samples_split=10`| 0.9472 | 0.0292 |
| **Random Forest** | No PCA | `n_estimators=100, max_depth=None, min_samples_split=2` | 0.9699 | 0.0146 |
| **Random Forest** | With PCA| `n_estimators=100, max_depth=None, min_samples_split=5` | 0.9687 | 0.0179 |
| **AdaBoost** | No PCA | `n_estimators=100, learning_rate=1` | 0.9844 | 0.0150 |
| **AdaBoost** | With PCA| `n_estimators=50, learning_rate=1` | 0.9811 | 0.0136 |
| **Gradient Boosting**| No PCA | `n_estimators=100, learning_rate=0.1, max_depth=3` | 0.9614 | 0.0121 |
| **Gradient Boosting**| With PCA| `n_estimators=100, learning_rate=0.1, max_depth=3` | 0.9579 | 0.0203 |
| **XGBoost** | No PCA | `n_estimators=100, max_depth=3, learning_rate=0.1` | 0.9772 | 0.0143 |
| **XGBoost** | With PCA| `n_estimators=100, max_depth=3, learning_rate=0.1` | 0.9667 | 0.0103 |
| **Stacking** | No PCA | Fixed config (SVM+KNN+RF; LR meta-learner) | 0.9807 | 0.0089 |
| **Stacking** | With PCA| Fixed config (SVM+KNN+RF; LR meta-learner) | 0.9774 | 0.0089 |

**Interpretation:** The best configuration is selected using the mean validation F1-score rather than by manually choosing a single parameter setting. The selected parameters also differ between No-PCA and PCA settings for several classifiers, showing that the preferred model configuration can depend on the feature representation.

## 10. Results

### 10.1 Independent Test-Set Metrics
**Table 3: Independent test-set performance for all models.**

| Model | Setting | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | No PCA | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| **SVM** | With PCA | 0.9649 | 0.9857 | 0.9583 | 0.9718 |
| **Naive Bayes** | No PCA | 0.9298 | 0.9444 | 0.9444 | 0.9444 |
| **Naive Bayes** | With PCA | 0.9211 | 0.9315 | 0.9444 | 0.9379 |
| **KNN** | No PCA | 0.9649 | 0.9595 | 0.9861 | 0.9726 |
| **KNN** | With PCA | 0.9561 | 0.9589 | 0.9722 | 0.9655 |
| **Logistic Regression**| No PCA | 0.9737 | 0.9726 | 0.9861 | 0.9793 |
| **Logistic Regression**| With PCA| 0.9737 | 0.9859 | 0.9722 | 0.9790 |
| **Decision Tree** | No PCA | 0.9386 | 0.9452 | 0.9583 | 0.9517 |
| **Decision Tree** | With PCA| 0.9035 | 0.9296 | 0.9167 | 0.9231 |
| **Random Forest** | No PCA | 0.9561 | 0.9589 | 0.9722 | 0.9655 |
| **Random Forest** | With PCA| 0.9298 | 0.9444 | 0.9444 | 0.9444 |
| **AdaBoost** | No PCA | 0.9561 | 0.9467 | 0.9861 | 0.9660 |
| **AdaBoost** | With PCA| 0.9474 | 0.9583 | 0.9583 | 0.9583 |
| **Gradient Boosting** | No PCA | 0.9561 | 0.9467 | 0.9861 | 0.9660 |
| **Gradient Boosting** | With PCA| 0.9561 | 0.9467 | 0.9861 | 0.9660 |
| **XGBoost** | No PCA | 0.9474 | 0.9459 | 0.9722 | 0.9589 |
| **XGBoost** | With PCA| 0.9474 | 0.9583 | 0.9583 | 0.9583 |
| **Stacking** | No PCA | 0.9737 | 0.9859 | 0.9722 | 0.9790 |
| **Stacking** | With PCA| 0.9561 | 0.9718 | 0.9583 | 0.9650 |

### 10.2 Five-Fold Cross-Validation
**Table 4: Fold-wise F1-scores for the selected configurations.**

| Model | Setting | F1-1 | F1-2 | F1-3 | F1-4 | F1-5 | Mean | Std. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | No PCA | 0.9661 | 0.9825 | 0.9825 | 0.9828 | 0.9913 | 0.9810 | 0.0082 |
| **SVM** | With PCA| 0.9739 | 0.9913 | 0.9913 | 0.9828 | 0.9825 | 0.9843 | 0.0065 |
| **Naive Bayes**| No PCA | 0.9310 | 0.9735 | 0.9123 | 0.9655 | 0.9565 | 0.9478 | 0.0228 |
| **Naive Bayes**| With PCA| 0.9244 | 0.9565 | 0.9310 | 0.9573 | 0.9474 | 0.9433 | 0.0134 |
| **KNN** | No PCA | 0.9828 | 0.9913 | 0.9649 | 0.9655 | 0.9825 | 0.9774 | 0.0104 |
| **KNN** | With PCA| 0.9828 | 0.9913 | 0.9649 | 0.9492 | 0.9828 | 0.9742 | 0.0152 |
| **Logistic Reg.**| No PCA | 0.9828 | 0.9825 | 0.9913 | 0.9828 | 0.9913 | 0.9861 | 0.0042 |
| **Logistic Reg.**| With PCA| 0.9828 | 0.9911 | 0.9913 | 0.9828 | 0.9825 | 0.9861 | 0.0042 |
| **Decision Tree**| No PCA | 0.9391 | 0.9455 | 0.9381 | 0.9402 | 0.9464 | 0.9418 | 0.0034 |
| **Decision Tree**| With PCA| 0.9180 | 0.9825 | 0.9310 | 0.9828 | 0.9217 | 0.9472 | 0.0292 |
| **Random Forest**| No PCA | 0.9735 | 0.9643 | 0.9464 | 0.9744 | 0.9911 | 0.9699 | 0.0146 |
| **Random Forest**| With PCA| 0.9483 | 0.9825 | 0.9474 | 0.9739 | 0.9913 | 0.9687 | 0.0179 |
| **AdaBoost** | No PCA | 0.9913 | 1.0000 | 0.9565 | 0.9828 | 0.9913 | 0.9844 | 0.0150 |
| **AdaBoost** | With PCA| 0.9661 | 1.0000 | 0.9825 | 0.9655 | 0.9913 | 0.9811 | 0.0136 |
| **Gradient Boost**|No PCA | 0.9649 | 0.9558 | 0.9464 | 0.9573 | 0.9825 | 0.9614 | 0.0121 |
| **Gradient Boost**|With PCA| 0.9310 | 0.9739 | 0.9369 | 0.9825 | 0.9649 | 0.9579 | 0.0203 |
| **XGBoost** | No PCA | 0.9913 | 0.9825 | 0.9558 | 0.9655 | 0.9911 | 0.9772 | 0.0143 |
| **XGBoost** | With PCA| 0.9573 | 0.9739 | 0.9550 | 0.9825 | 0.9649 | 0.9667 | 0.0103 |
| **Stacking** | No PCA | 0.9828 | 0.9825 | 0.9643 | 0.9828 | 0.9913 | 0.9807 | 0.0089 |
| **Stacking** | With PCA| 0.9744 | 0.9825 | 0.9649 | 0.9739 | 0.9913 | 0.9774 | 0.0089 |

### 10.3 Main No-PCA versus PCA Comparison
**Table 5: Overall comparison of No-PCA and PCA configurations.**

| Model | Acc. NP | Acc. P | F1 NP | F1 P | $\Delta$F1 | CV NP | CV P | Std NP | Std P |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.9825 | 0.9649 | 0.9861 | 0.9718 | -0.0143 | 0.9810 | 0.9843 | 0.0082 | 0.0065 |
| **Naive Bayes**| 0.9298 | 0.9211 | 0.9444 | 0.9379 | -0.0065 | 0.9478 | 0.9433 | 0.0228 | 0.0134 |
| **KNN** | 0.9649 | 0.9561 | 0.9726 | 0.9655 | -0.0071 | 0.9774 | 0.9742 | 0.0104 | 0.0152 |
| **Logistic Reg**| 0.9737 | 0.9737 | 0.9793 | 0.9790 | -0.0003 | 0.9861 | 0.9861 | 0.0042 | 0.0042 |
| **Decision Tree**| 0.9386 | 0.9035 | 0.9517 | 0.9231 | -0.0286 | 0.9418 | 0.9472 | 0.0034 | 0.0292 |
| **Random Forest**| 0.9561 | 0.9298 | 0.9655 | 0.9444 | -0.0211 | 0.9699 | 0.9687 | 0.0146 | 0.0179 |
| **AdaBoost** | 0.9561 | 0.9474 | 0.9660 | 0.9583 | -0.0077 | 0.9844 | 0.9811 | 0.0150 | 0.0136 |
| **Gradient Bst**| 0.9561 | 0.9561 | 0.9660 | 0.9660 | +0.0000 | 0.9614 | 0.9579 | 0.0121 | 0.0203 |
| **XGBoost** | 0.9474 | 0.9474 | 0.9589 | 0.9583 | -0.0006 | 0.9772 | 0.9667 | 0.0143 | 0.0103 |
| **Stacking** | 0.9737 | 0.9561 | 0.9790 | 0.9650 | -0.0140 | 0.9807 | 0.9774 | 0.0089 | 0.0089 |

**Key Observations:**
* PCA did not increase the independent test F1-score for any model. Gradient Boosting was unchanged, while all other models decreased.
* The largest test F1 decrease occurred for Decision Tree, falling from 0.9517 to 0.9231.
* SVM showed the largest positive change in cross-validation mean F1.
* Naïve Bayes experienced the largest reduction in CV standard deviation (improved fold stability).

## 11. Result Visualization
> *(Note: Placeholders for generated graphs)*
* **Figure 4:** Test-set F1-score comparison.
* **Figure 5:** Test-set accuracy comparison between No PCA and With PCA.
* **Figure 6:** Confusion matrix for the best PCA model, Logistic Regression.
* **Figure 7:** ROC curve for the best PCA Logistic Regression model (ROC-AUC: 0.9954).
* **Figure 8:** Precision-Recall curve for the best PCA Logistic Regression model (Average Precision: 0.9971).
* **Figure 9:** Five-fold cross-validation F1 standard deviation for No-PCA and PCA configurations.

## 12. Discussion

### 12.1 Which Models Improved Most with PCA?
Based on independent test-set F1-score, no classifier improved after PCA. Gradient Boosting was the only model whose F1-score remained unchanged at 0.9660. The behaviour is consistent with the nature of the algorithms. SVM, KNN, and Logistic Regression depend more directly on the geometry of the feature space. Tree-based models make threshold decisions on individual features, so replacing original variables with principal components can remove feature-specific information that is useful for their splits.

### 12.2 Did PCA Reduce Variance Across Folds?
PCA did not universally reduce validation variability. It reduced the standard deviation for SVM, Naïve Bayes, and Logistic Regression. For the remaining models, the standard deviation increased.

### 12.3 Was PCA Beneficial for Reducing Overfitting?
The experimental results do not provide evidence that PCA universally reduced overfitting. Several models had lower test F1 after PCA, even when their cross-validation mean F1 was similar or slightly higher. Dimensionality reduction alone is not sufficient to guarantee improved generalization.

### 12.4 Linear Models versus Ensemble Models
Logistic Regression was particularly robust. Among ensemble models, Random Forest, AdaBoost, and XGBoost generally declined after PCA. This suggests that the original feature representation contains useful structure that nonlinear tree ensembles can exploit directly.

### 12.5 Stacking Robustness
Stacking remained relatively stable in terms of fold variability, but its independent test performance was meaningfully reduced by PCA.

### 12.6 Best Models
The best No-PCA individual model was **SVM** with test F1-score 0.9861 and accuracy 0.9825. The best PCA individual model was **Logistic Regression** with test F1-score 0.9790 and accuracy 0.9737. 

## 13. Conclusion
The experiment demonstrated the effect of PCA-based dimensionality reduction on a set of machine-learning classifiers using the Breast Cancer Wisconsin Diagnostic dataset. The feature space was reduced from 30 original variables to 10 principal components while retaining 95.2677% of the variance. 

The results show that PCA does not automatically improve classification performance. The best No-PCA model was SVM (F1-score 0.9861), whereas the best PCA model was Logistic Regression (F1-score 0.9790). Overall, PCA should be regarded as a data-dependent dimensionality-reduction technique rather than a transformation that universally improves predictive performance.

## 14. References
1. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
2. I. T. Jolliffe and J. Cadima, “Principal component analysis: a review and recent developments,” *Philosophical Transactions of the Royal Society A*, vol. 374, no. 2065, 2016.
3. T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference*, 2016, pp. 785–794.
4. L. Breiman, “Random Forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001.