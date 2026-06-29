# Credit-Scoring-Project
This project focuses on building a high-performance credit scoring model to predict the probability of client default.

## Tech Stack
* Data Processing: Dask (for handling datasets larger than RAM), Pandas, NumPy
* Machine Learning: CatBoost, Scikit-learn
* Validation & Tuning: RandomizedSearchCV (5-fold Cross-Validation)

## Data Overview & Feature Dictionary
The model was trained on a large-scale, pre-binned, and anonymized banking dataset. Below is the detailed feature dictionary:

| Feature Name | Format / Type | Description |
| :--- | :--- | :--- |
| id | ID / Integer | Application identifier. Applications are numbered sequentially; a higher number corresponds to a later application date. |
| rn | Integer | Sequential number of the credit product in the client's credit history. A higher number corresponds to a product with a later opening date. |
| pre_since_opened | Binarized* | Number of days from the credit opening date to the data collection date. |
| pre_since_confirmed | Binarized* | Number of days from the credit information confirmation date to the data collection date. |
| pre_pterm | Binarized* | Scheduled number of days from the credit opening date to the closing date. |
| pre_fterm | Binarized* | Actual number of days from the credit opening date to the closing date. |
| pre_till_pclose | Binarized* | Scheduled number of days from the data collection date to the credit closing date. |
| pre_till_fclose | Binarized* | Actual number of days from the data collection date to the credit closing date. |
| pre_loans_credit_limit | Binarized* | Credit limit. |
| pre_loans_next_pay_summ | Binarized* | Next credit payment amount. |
| pre_loans_outstanding | Binarized* | Outstanding (unpaid) balance of the credit. |
| pre_loans_total_overdue | Binarized* | Current overdue balance. |
| pre_loans_max_overdue_sum | Binarized* | Maximum overdue balance. |
| pre_loans_credit_cost_rate | Binarized* | Total cost of credit (annual percentage rate). |
| pre_loans5 | Binarized* | Number of delinquencies lasting up to 5 days. |
| pre_loans530 | Binarized* | Number of delinquencies lasting from 5 to 30 days. |
| pre_loans3060 | Binarized* | Number of delinquencies lasting from 30 to 60 days. |
| pre_loans6090 | Binarized* | Number of delinquencies lasting from 60 to 90 days. |
| pre_loans90 | Binarized* | Number of delinquencies lasting more than 90 days. |
| is_zero_loans_5 | Flag (0/1) | Indicates no delinquencies lasting up to 5 days. |
| is_zero_loans_530 | Flag (0/1) | Indicates no delinquencies lasting from 5 to 30 days. |
| is_zero_loans_3060 | Flag (0/1) | Indicates no delinquencies lasting from 30 to 60 days. |
| is_zero_loans_6090 | Flag (0/1) | Indicates no delinquencies lasting from 60 to 90 days. |
| is_zero_loans90 | Flag (0/1) | Indicates no delinquencies lasting more than 90 days. |
| pre_util | Binarized* | Ratio of outstanding balance to the credit limit. |
| pre_over2limit | Binarized* | Ratio of current overdue balance to the credit limit. |
| pre_maxover2limit | Binarized* | Ratio of maximum overdue balance to the credit limit. |
| is_zero_util | Flag (0/1) | Indicates that the ratio of outstanding balance to the credit limit equals 0. |
| is_zero_over2limit | Flag (0/1) | Indicates that the ratio of current overdue balance to the credit limit equals 0. |
| is_zero_maxover2limit | Flag (0/1) | Indicates that the ratio of maximum overdue balance to the credit limit equals 0. |
| enc_paym_{0..N} | Encoded** | Monthly payment statuses for the last N months. |
| enc_loans_account_holder_type | Encoded** | Account holder relationship type to the loan. |
| enc_loans_credit_status | Encoded** | Loan / credit status. |
| enc_loans_account_cur | Encoded** | Loan currency. |
| enc_loans_credit_type | Encoded** | Loan / credit type. |
| pclose_flag | Flag (0/1) | Scheduled number of days from loan opening to closing is undefined. |
| fclose_flag | Flag (0/1) | Actual number of days from loan opening to closing is undefined. |

\* Binarization / Binning: The range of values for this field is divided into N non-overlapping intervals. Each interval is randomly assigned a unique number from 0 to N-1, and the original value is replaced by the interval number it belongs to.  
\*\* Encoded: Categorical features that have been label-encoded or categorical-encoded for privacy protection and model optimization.

## Core Challenges & Key Features

**The core challenges** involved:
*   Managing memory efficiency using Dask for "lazy" computation.
*   Processing large-scale binned and coded banking data.
*   Engineering new features to capture credit discipline and debt burden.

**Key Features:**
*   Scalable Architecture: Used Dask for "lazy" computing to process data in chunks without memory overflows.
*   Feature Engineering:
    Delinquency Buckets: Categorized overdue periods (e.g., <5d, 5-30d, 30-60d, 90d+).
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
