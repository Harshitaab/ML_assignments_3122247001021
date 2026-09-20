Absolutely. Based on the **Experiment 5 results you provided**, here is a concise, GitHub-ready `README.md`.

# Experiment 5 – Decision Tree and Random Forest Classification

## Aim

To implement and compare **Decision Tree** and **Random Forest** classification models using hyperparameter tuning and 5-fold cross-validation.

## Objectives

* Implement a Decision Tree classifier.
* Implement a Random Forest classifier.
* Tune model hyperparameters using 5-fold cross-validation.
* Compare model performance using Accuracy, Precision, Recall, F1 Score and ROC-AUC.
* Analyze the performance of both models on a held-out test set.

## Algorithms Used

1. Decision Tree Classifier
2. Random Forest Classifier

## Tools and Technologies

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab

## Hyperparameter Tuning

### Decision Tree

The Decision Tree was tuned using:

* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

**Best Parameters:**

```text
criterion = gini
max_depth = 5
min_samples_split = 2
min_samples_leaf = 4
```

**Best Mean CV F1 Score:** `0.95058`

### Random Forest

The Random Forest was tuned using:

* `n_estimators`
* `max_depth`
* `max_features`
* `bootstrap`

**Best Parameters:**

```text
n_estimators = 100
max_depth = None
max_features = log2
bootstrap = True
```

**Best Mean CV F1 Score:** `0.97180`

## 5-Fold Cross-Validation Results

| Model         | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |    Std |
| ------------- | -----: | -----: | -----: | -----: | -----: | ------: | -----: |
| Decision Tree | 0.9451 | 0.9451 | 0.9121 | 0.9231 | 0.9670 |  0.9385 | 0.0192 |
| Random Forest | 0.9670 | 0.9780 | 0.9341 | 0.9560 | 0.9890 |  0.9648 | 0.0189 |

## Final Test Set Results

| Model         |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Decision Tree |     0.9035 |     0.9420 |     0.9028 |     0.9220 |     0.9358 |
| Random Forest | **0.9561** | **0.9589** | **0.9722** | **0.9655** | **0.9924** |

## Result

The tuned Decision Tree achieved a mean 5-fold CV accuracy of **93.85%**, while the Random Forest achieved **96.48%**.

On the held-out test set, the Random Forest obtained:

* **Accuracy:** 95.61%
* **Precision:** 95.89%
* **Recall:** 97.22%
* **F1 Score:** 96.55%
* **ROC-AUC:** 99.24%

## Conclusion

The experiment demonstrates the use of Decision Tree and Random Forest classifiers with hyperparameter tuning and 5-fold cross-validation. The obtained results show the performance of both models across cross-validation and held-out test evaluation.

## Project Structure

```text
Experiment-5/
│
├── Experiment_5.ipynb
├── README.md
│
├── figures/
│   ├── decision_tree_cv_results.csv
│   ├── random_forest_cv_results.csv
│   └── five_fold_accuracy_comparison.csv
│
└── results/
    └── final_results.csv
```

## Evaluation Metrics

* **Accuracy:** Overall proportion of correctly classified samples.
* **Precision:** Proportion of predicted positive samples that are actually positive.
* **Recall:** Proportion of actual positive samples correctly identified.
* **F1 Score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Measures the model's ability to distinguish between classes.

