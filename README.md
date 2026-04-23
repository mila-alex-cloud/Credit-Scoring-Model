# Credit-Scoring-Project
This project focuses on building a high-performance credit scoring model to predict the probability of client default.

**The core challenges** involved:
*   Managing memory efficiency using Dask for "lazy" computation.
*   Processing large-scale binned and coded banking data.
*   Engineering new features to capture credit discipline and debt burden.

**Tech Stack:**
*   Data Processing: Dask (for handling datasets larger than RAM)
*   Machine Learning: CatBoost, Scikit-learn

**Key Features:**
*   Scalable Architecture: Used Dask for "lazy" computing to process data in chunks without memory overflows.
*   Feature Engineering:
    Delinquency Buckets: Categorized overdue periods (e.g., <5d, 5-30d, 90d+).
    Credit Dynamics: Age of oldest credit, limit utilization rates, and total debt burden.
    Clean Logic: Removed data contradictions (e.g., removing >90d delinquencies not yet flagged as defaults).
*   Hyperparameter Tuning: Performed RandomizedSearchCV with 5-fold cross-validation to optimize model stability and ROC-AUC score.

**The Results**
The CatBoost model is recommended for deployment because:

*   Native Categorical Support: It handles categorical features internally, eliminating the need for complex manual preprocessing (like One-Hot Encoding).
*   Superior Performance: It achieved the highest ROC-AUC (0.77) and Gini coefficient (0.54) among all tested models.

Alternative Consideration: Logistic Regression

*   Interpretability: LogReg remains the most transparent and interpretable model for stakeholders.
*   Selection Logic: Had the Logistic Regression model shown a comparable ROC-AUC, it would have been the preferred choice due to its simplicity. However, the performance gap justifies the use of the more complex CatBoost model.

**The Project contains extended results interpretation, featuring Confusion Matrix and Classification Report (with both default and custom thresholds).**
