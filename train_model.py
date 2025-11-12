import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heartsense_pipeline.pkl")

def load_data(path=DATA_PATH):
    """Load the heart dataset"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Could not find dataset at: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def build_and_train(df):
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Class balance: {(y.value_counts(normalize=True) * 100).round(1).to_dict()}%")

    # Define feature types
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Build preprocessing pipeline
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Create model with regularization to prevent overfitting
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,          # Number of trees
            max_depth=8,               # Limit tree depth to prevent overfitting
            min_samples_split=15,      # Require more samples to split a node
            min_samples_leaf=6,        # Require more samples in leaf nodes
            max_features='sqrt',       # Limit features considered at each split
            random_state=42,
            class_weight="balanced",   # Handle class imbalance
            n_jobs=-1
        ))
    ])

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Perform cross-validation
    print("\n" + "="*60)
    print("🔄 CROSS-VALIDATION (5-Fold)")
    print("="*60)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"CV ROC AUC scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Mean CV ROC AUC: {cv_scores.mean():.3f} (± {cv_scores.std() * 2:.3f})")

    # Train final model
    print("\n⏳ Training final model...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete!")

    # Evaluate on training set
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    # Display results
    print("\n" + "="*60)
    print("📈 TRAINING SET PERFORMANCE")
    print("="*60)
    print(f"Accuracy:  {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"ROC AUC:   {train_auc:.3f}")

    print("\n" + "="*60)
    print("🎯 TEST SET PERFORMANCE (Unseen Data)")
    print("="*60)
    print(f"Accuracy:  {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"ROC AUC:   {test_auc:.3f}")
    
    # Check overfitting
    overfit_gap = train_acc - test_acc
    print(f"\n📊 Overfitting Analysis:")
    print(f"   Gap between train and test: {overfit_gap:.3f}")
    if overfit_gap > 0.1:
        print("   ⚠️  WARNING: Model may be overfitting (gap > 0.1)")
    elif overfit_gap > 0.05:
        print("   ⚠️  Slight overfitting detected (gap > 0.05)")
    else:
        print("   ✅ Good generalization - minimal overfitting!")

    # Detailed test set results
    print("\n" + "="*60)
    print("📋 DETAILED TEST SET RESULTS")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Disease (0)', 'Has Disease (1)']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n                 Predicted")
    print(f"                 No    Yes")
    print(f"Actual  No     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        Yes    {cm[1,0]:4d}  {cm[1,1]:4d}")
    print(f"\nTrue Negatives:  {cm[0,0]} (correctly identified healthy)")
    print(f"False Positives: {cm[0,1]} (false alarms)")
    print(f"False Negatives: {cm[1,0]} (missed disease cases) ⚠️ CRITICAL")
    print(f"True Positives:  {cm[1,1]} (correctly identified disease)")

    # Feature importance
    try:
        feature_names = (numeric_features + 
                        list(pipeline.named_steps['preproc']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_features)))
        
        importances = pipeline.named_steps['clf'].feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*60)
        print("🔝 TOP 10 MOST IMPORTANT FEATURES")
        print("="*60)
        for idx, row in feature_imp_df.head(10).iterrows():
            print(f"{row['feature']:20s} {row['importance']:.4f} {'█' * int(row['importance'] * 100)}")
    except Exception as e:
        print(f"\n⚠️  Could not extract feature importance: {e}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    
    print("\n" + "="*60)
    print("💾 MODEL SAVED")
    print("="*60)
    print(f"Location: {MODEL_PATH}")
    print(f"File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    print("\n✅ Model is ready to use with the Flask app!")
    print("   Run: python app.py")

    return pipeline

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🫀 HEARTSENSE MODEL TRAINING")
    print("="*60)
    df = load_data()
    pipeline = build_and_train(df)
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)