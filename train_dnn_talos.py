import os
import pandas as pd
import numpy as np
import joblib
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import talos
from talos import Scan
from talos.utils import lr_normalizer

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heartsense_dnn.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# Prevent retraining if model already exists
# --------------------------------------------------
if os.path.exists(MODEL_PATH):
    print("✅ Model already trained.")
    print("📦 Model path:", MODEL_PATH)
    print("🚫 Training skipped.")
    print("👉 Delete the model file if you want to retrain.")
    exit()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"].astype(int)

# -------------------------------
# Train / Validation / Test split
# -------------------------------

# Split 1: Train+Val and Test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split 2: Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# -------------------------------
# Scaling (REQUIRED for DNN)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scaler for Flask
joblib.dump(scaler, os.path.join(MODEL_DIR, "dnn_scaler.pkl"))

# -------------------------------
# Convert to NumPy (Talos FIX)
# -------------------------------
X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)

y_train = np.asarray(y_train).reshape(-1, 1)
y_val = np.asarray(y_val).reshape(-1, 1)
y_test = np.asarray(y_test).reshape(-1, 1)


# --------------------------------------------------
# Model definition (Talos expects this format)
# --------------------------------------------------
from tensorflow.keras.callbacks import ModelCheckpoint

def heart_dnn_model(X_train, y_train, X_val, y_val, params):

    model = Sequential()

    model.add(Dense(params['units1'], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(params['dropout']))

    model.add(Dense(params['units2'], activation='relu'))
    model.add(Dropout(params['dropout']))

    model.add(Dense(params['units3'], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr_normalizer(params['lr'], Adam)),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[checkpoint],
        verbose=0
    )

    return history, model


# --------------------------------------------------
# Talos hyperparameter space (paper-aligned)
# --------------------------------------------------
p = {
    'units1': [64, 128],
    'units2': [32, 64],
    'units3': [16, 32],
    'dropout': [0.2, 0.3],
    'lr': [0.001, 0.0005],
    'batch_size': [16, 32],
    'epochs': [50, 100]
}

# --------------------------------------------------
# Run Talos Scan
# --------------------------------------------------
scan = Scan(
    x=X_train,
    y=y_train,
    model=heart_dnn_model,
    params=p,
    experiment_name='HeartSense_DNN',
    reduction_metric='val_accuracy',
    minimize_loss=False
)


print("\n💾 DNN model saved at:")
print(MODEL_PATH)
print("\n✅ DNN training complete!")

print("\n✅ Talos training completed.")
print("💾 Best model saved automatically at:")