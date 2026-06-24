import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load scaler
scaler = joblib.load("dnn_scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# Load DNN model
model = load_model("heartsense_dnn.h5")

# Predictions
y_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== DNN RESULTS =====")
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))