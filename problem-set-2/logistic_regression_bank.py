import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ============================================================
# 1. Load Dataset
# ============================================================

DATA_PATH = "/kaggle/input/datasets/taimurislamtahnaf/bank-data/bank-data/bank-full.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["y"].value_counts())


# ============================================================
# 2. Prepare Features and Target
# ============================================================

X = df.drop("y", axis=1)
y = df["y"]

# Convert categorical features into numerical dummy variables
X = pd.get_dummies(X, drop_first=True)

# Encode target: no = 0, yes = 1
le = LabelEncoder()
y = le.fit_transform(y)


# ============================================================
# 3. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 5. Train Logistic Regression Model
# ============================================================

model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# 6. Predictions and Accuracy
# ============================================================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 7. Classification Report
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, test_pred))


# ============================================================
# 8. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, test_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 9. ROC Curve and AUC
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ============================================================
# 10. Save Model and Preprocessing Objects
# ============================================================

joblib.dump(model, "logistic_regression_bank_model.pkl")
joblib.dump(scaler, "bank_scaler.pkl")
joblib.dump(X.columns.tolist(), "bank_feature_columns.pkl")

print("\nModel and preprocessing objects saved successfully!")


# ============================================================
# 11. Test a Sample from the Test Set
# ============================================================

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Sample Prediction: Customer will subscribe (YES)")
else:
    print("Sample Prediction: Customer will NOT subscribe (NO)")


# ============================================================
# 12. Predict for a New Customer
# ============================================================

new_customer = pd.DataFrame({
    "age": [35],
    "job": ["technician"],
    "marital": ["married"],
    "education": ["secondary"],
    "default": ["no"],
    "balance": [2500],
    "housing": ["yes"],
    "loan": ["no"],
    "contact": ["cellular"],
    "day": [15],
    "month": ["may"],
    "duration": [180],
    "campaign": [2],
    "pdays": [-1],
    "previous": [0],
    "poutcome": ["unknown"]
})

new_customer = pd.get_dummies(new_customer)
new_customer = new_customer.reindex(columns=X.columns, fill_value=0)
new_customer = scaler.transform(new_customer)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("New Customer Prediction: Customer will Subscribe (YES)")
else:
    print("New Customer Prediction: Customer will NOT Subscribe (NO)")
