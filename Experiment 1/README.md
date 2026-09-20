# Exploratory Data Analysis on the Iris Dataset 🌸

**Course:** ICS1512 - Machine Learning Algorithms Laboratory (Experiment 1)  
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Author:** Harshitaa B 

---

## 📌 Project Objective
The objective of this project is to perform a comprehensive **Exploratory Data Analysis (EDA)** on the Iris dataset using statistical summaries and visual representations. The project aims to:
* Identify data distributions and detect missing values.
* Evaluate potential outliers across continuous attributes.
* Assess mathematical model assumptions (normality, linearity, multicollinearity).
* Guide the choice of feature scaling techniques and machine learning algorithms for subsequent modeling experiments.

## 📊 Dataset Description
The Iris dataset is a multivariate dataset consisting of morphological measurements of three species of the Iris flower.

| Feature | Description |
| :--- | :--- |
| **Source** | UCI Machine Learning Repository (`iris.csv`) |
| **Samples** | 150 (50 instances per class) |
| **Features** | 4 (sepal length, sepal width, petal length, petal width) - measured in cm |
| **Classes** | 3 (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |
| **Missing Values** | 0 (Dataset is completely clean) |

## 🔍 Key EDA Findings

### 1. Missing Values & Class Distribution
* **Missing Data:** Visualized via heatmaps and confirmed via statistical checks; there are **zero** missing entries. No imputation or row deletion is required.
* **Balance:** The dataset is perfectly balanced with exactly 50 samples for each of the three species. 

### 2. Feature Distributions
* **Sepal Width:** Approximately normal (bell-shaped) distribution (skewness = 0.33), making it a good candidate for Standardization ($Z$-score scaling).
* **Petal Length & Width:** Both exhibit strongly bimodal distributions, clearly separating `Iris-setosa` (which has much smaller petals) from the other two species.
* **Sepal Length:** Mildly bimodal, hinting at underlying class structures.

### 3. Correlation & Multicollinearity
* **Strong Positive Correlation:** Petal length and petal width are almost perfectly correlated ($r = 0.96$). Sepal length also correlates strongly with both petal length ($r = 0.87$) and petal width ($r = 0.82$).
* **Linear Separability:** Pair plots reveal that `Iris-setosa` is linearly separable from the other two classes on almost every feature pair. `Iris-versicolor` and `Iris-virginica` show moderate overlap.

### 4. Outlier Detection
Using the Interquartile Range (IQR) method:
* **Sepal Width:** Identified 4 potential outliers. These were retained as they fall within biologically plausible ranges and are not data entry errors.
* **Other Features:** 0 outliers detected.

## ⚙️ Planned Data Preprocessing
Based on the EDA findings, the following preprocessing steps are recommended before model training:
1. **Encoding:** Apply `LabelEncoder` to the target categorical column (`class`).
2. **Feature Scaling:** Standard scaling (Standardization: $z = \frac{x - \mu}{\sigma}$) is recommended across all features to handle multicollinearity and prepare the data for distance-based or linear models.
3. **Data Splitting:** An 80:20 stratified train-test split is planned to maintain the perfect class balance.

## 💻 Experimental Setup
* **Language:** Python 3.12.13
* **Libraries Used:** Pandas, Scikit-Learn, Matplotlib/Seaborn (for visualization)
* **Environment:** Google Colab / Windows PC

## 📚 Theoretical Concepts Explored
This laboratory experiment also extensively reviewed foundational data science concepts:
* Standardization vs. Normalization (Min-Max Scaling).
* Mathematical techniques for Outlier Detection (Z-score, IQR, Mahalanobis Distance, LOF).
* Missing value imputation strategies (MCAR, MAR, MNAR).
* Feature selection using Correlation Coefficients and Mutual Information.
* Validating model assumptions (Linearity, Normality, Homoscedasticity).
