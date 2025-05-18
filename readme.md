# Project: Adult Income Prediction using Machine Learning

**Course:** DSAA2011 Machine Learning

**Authors:**
* Hua XU (hxu401@connect.hkust-gz.edu.cn)
* Jianhao RUAN (jruan189@connect.hkust-gz.edu.cn)
* Leyi WU (lwu398@connect.hkust-gz.edu.cn)

## 1. Introduction

This project addresses the binary classification problem of predicting whether an individual's annual income exceeds $50,000 based on various attributes. We utilized the Census Income dataset (also known as the "Adult" dataset from UCI Machine Learning Repository [1]).

The project encompasses a comprehensive machine learning pipeline, starting from data preprocessing, exploratory data analysis and visualization (including t-SNE), clustering to uncover latent data structures, feature importance analysis, and finally, the training and evaluation of multiple classification models. Key challenges identified include the high dimensionality resulting from one-hot encoding, poor initial separability of income classes, the presence of sub-clusters within classes, significant feature redundancy, and pronounced class imbalance in the dataset.

## 2. Dataset

* **Source:** Census Income dataset [1] (Adult dataset, UCI Machine Learning Repository).
* **Variables:** The dataset comprises 14 usable variables, including 6 integer types and 8 categorical types (e.g., age, education, work class, occupation, hours-per-week).
* **Target Variable:** Binary classification - whether income is `<=50K` or `>50K`.
* **Class Imbalance:** The dataset exhibits significant class imbalance, with 37,155 instances labeled `<=50K` and 11,687 instances labeled `>50K`.

## 3. Methodology

Our analytical pipeline involved the following key stages:

### 3.1. Data Preprocessing
* **Categorical Variables:**
    * Missing values were imputed using the most frequent value for each feature.
    * Features were transformed into numerical format using one-hot encoding (via scikit-learn's `OneHotEncoder`).
* **Numeric Variables:**
    * Missing values were replaced with the median value of the respective variable.
    * Features were standardized to have a mean of 0 and a variance of 1 (using scikit-learn's `StandardScaler`).
* **Outcome:** After preprocessing, the dataset expanded to 105 features.

### 3.2. Data Visualization & Exploratory Analysis
* **t-SNE (t-distributed Stochastic Neighbor Embedding):** Used to project the high-dimensional data into 2D for visualization. A perplexity of 50 was found to strike a good balance.
* **Observations:**
    * High dimensionality was evident.
    * Income classes showed poor separability in the low-dimensional embedding.
    * Potential sub-clusters were observed within each class, suggesting heterogeneity.
    * Samples with income `<50K` were broadly distributed, while those with `>50K` were more concentrated.
    * Many features appeared non-discriminative, suggesting redundancy, which was later confirmed by feature importance analysis.

### 3.3. Clustering Analysis
To understand underlying data groupings, clustering was performed on the 3D t-SNE embedded data using:
* **K-Means Clustering:** Implemented with k-means++ initialization.
* **Agglomerative Hierarchical Clustering:** Using Ward's method.
* **Findings:**
    * K-Means achieved better internal cohesion metrics (Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score).
    * Agglomerative clustering aligned more closely with the true income labels (higher Adjusted Rand Score), indicating complex, non-convex patterns in the data that violate K-Means' spherical cluster assumption.

### 3.4. Classification Models
A suite of classifiers was trained and evaluated:
* **Logistic Regression:** Included as a baseline.
    * Parameters: `class_weight='balanced'`, `max_iter=1000`, `random_state=42`, `n_jobs=-1`.
* **Decision Tree:**
    * Parameters: `max_depth=15`, `class_weight='balanced'`, `random_state=42`.
* **Random Forest:** An ensemble method.
    * Parameters: `n_estimators=200`, `max_depth=15`, `min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
* **XGBoost:** A gradient boosting ensemble method.
    * The final model, achieving an AUC of 0.929 (as detailed in Section 4), used the following key parameters: `n_estimators=200`, `max_depth=15`.
    * Other relevant parameters, consistent with initial explorations or common practices for such models, included: `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight` (adjusted to handle class imbalance), `eval_metric='auc'`, `random_state=42`, `n_jobs=-1`.

### 3.5. Handling Class Imbalance
Two resampling strategies were investigated to address the pronounced class imbalance:
* **Undersampling:** Sampling 8,100 instances from each class.
* **Oversampling (Synthetic Data):** Oversampling the minority class to 26,000 instances by interpolating between samples.
* **Findings:** Both approaches were found to effectively mitigate bias towards the majority class and improve robustness, particularly recall on high-income samples.

## 4. Results and Key Findings

* **Model Performance:** Ensemble methods, particularly XGBoost, consistently outperformed simpler models. The XGBoost model (configured with a depth of 15 and 200 trees) achieved the highest Area Under the ROC Curve (AUC).
    * XGBoost: AUC = 0.929
    * Random Forest: AUC = 0.913
    * Logistic Regression: AUC = 0.904
    * Decision Tree (vanilla): AUC = 0.867
* **Feature Importance:** Analysis revealed that marital status ("married") was a highly important feature. XGBoost tended to focus on a few key features, while standard Decision Trees and Random Forests distributed importance more broadly.
* **Decision Boundaries:** Visualizations (using both t-SNE and PCA projections) showed that tree-based models, especially ensembles, were capable of defining more complex and precise decision boundaries compared to logistic regression.
* **Impact of Tree Depth:** Increasing the depth of vanilla decision trees initially improved accuracy but eventually led to overfitting. For ensemble models, the report noted that the initially chosen hyperparameters might have been too large, potentially causing overfitting, and suggested that optimal depths for ensemble trees are often less than 10.

## 5. Conclusion

This project successfully demonstrated a comprehensive pipeline for the binary classification of adult income. Key takeaways include:

1.  **Feature Redundancy:** A significant number of original features in the Census Income dataset are redundant and could benefit from pruning or feature engineering.
2.  **Model Suitability:** Non-linear, tree-based ensemble models like XGBoost are well-suited to the complex data distribution observed.
3.  **Importance of Sampling:** Addressing class imbalance through techniques like under-sampling or over-sampling is crucial for robust performance and fair evaluation on socio-economic datasets.

The most effective solution identified was XGBoost combined with appropriate sampling techniques.

## 6. Future Work

* Exploration of advanced feature selection methods.
* Investigation of alternative remedies for class imbalance (e.g., cost-sensitive learning).
* Extending the analytical framework to other socio-economic datasets.

## 7. Credits

* **Hua XU:** Drafted the report and coordinated team members.
* **Jianhao RUAN:** Data visualization, organized the coding part, and polished the report.
* **Leyi WU:** Responsible for data processing, model training, and polished the report.
* **AI Assistance:** Used for formatting LaTeX tables and figures, polishing the report (with all arguments made by the team), drafting the Introduction section, and Q&A on in-class content.

## 8. References

[1] Becker, B., & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
* Foundational knowledge and course materials, including lecture slides by Prof. ZHONG and Prof. YANG from DSAA2011, were instrumental to this project.