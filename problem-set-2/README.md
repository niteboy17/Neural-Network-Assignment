# Bank Marketing Subscription Prediction

## Project Overview

This project uses **Logistic Regression** to predict whether a bank customer will subscribe to a term deposit based on demographic, financial, and campaign-related information.

The dataset is the **Bank Marketing Dataset** (`bank-full.csv`), containing information about customers contacted during a bank marketing campaign.

## Dataset

- **Dataset:** Bank Marketing (`bank-full.csv`)
- **Rows:** 45,211
- **Columns:** 17
- **Target variable:** `y`
  - `no` → Customer did not subscribe
  - `yes` → Customer subscribed

### Main Features

- Age
- Job
- Marital status
- Education
- Default status
- Account balance
- Housing loan
- Personal loan
- Contact type
- Contact day/month
- Call duration
- Campaign contacts
- Days since previous contact
- Number of previous contacts
- Previous campaign outcome

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the CSV dataset using Pandas.
2. Separated the target variable `y` from the input features.
3. Converted categorical variables into numerical values using `pd.get_dummies()`.
4. Encoded the target variable using `LabelEncoder`.
5. Split the dataset into:
   - 80% training data
   - 20% testing data
6. Standardized the numerical feature space using `StandardScaler`.

## Model

A **Logistic Regression** classifier was used with the following configuration:

```python
LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=42
)
```

## Results

### Accuracy

| Metric | Score |
|---|---:|
| Training Accuracy | **90.26%** |
| Test Accuracy | **89.87%** |
| ROC AUC | **0.905** |

The model achieved a **test accuracy of 89.87%**.

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.92      0.98      0.94      7952
           1       0.65      0.34      0.45      1091

    accuracy                           0.90      9043
   macro avg       0.78      0.66      0.70      9043
weighted avg       0.88      0.90      0.88      9043
```

### Confusion Matrix

```text
[[7755  197]
 [ 719  372]]
```

### ROC Curve

The ROC curve produced an **AUC of 0.905**, indicating that the model has good ability to distinguish between customers who subscribe and those who do not.

## Example Prediction

A sample customer was tested using the trained model.

```text
Prediction: Customer will NOT subscribe (No)
```

A new customer with the following profile was also tested:

- Age: 35
- Job: Technician
- Marital Status: Married
- Education: Secondary
- Balance: 2500
- Housing Loan: Yes
- Personal Loan: No
- Contact: Cellular
- Call Duration: 180 seconds
- Campaign: 2

Prediction:

```text
Customer will NOT Subscribe (NO)
```

## Files

```text
.
├── logistic_regression_bank.py
├── logistic_regression_bank_model.pkl
├── bank_scaler.pkl
├── bank_feature_columns.pkl
└── README.md
```

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib joblib
```

Run the Python script:

```bash
python logistic_regression_bank.py
```

> **Note:** Update `DATA_PATH` in the Python file if the dataset is stored in a different location.

## Conclusion

The Logistic Regression model achieved a **89.87% test accuracy** and a **0.905 ROC AUC score** on the Bank Marketing dataset. The results show that Logistic Regression can provide a strong baseline for predicting customer subscription behavior, although the lower recall for the positive class indicates that there is room for improvement in identifying customers who actually subscribe.
