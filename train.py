from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

from preprocess import load_and_preprocess_data

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains Random Forest and XGBoost classifiers on the preprocessed NSL-KDD dataset.
    Prints accuracy and full classification report for both models.
    Saves the Random Forest model and prints the top 15 most important features.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        
    Returns:
        rf_model: Trained Random Forest classifier.
        xgb_model: Trained XGBoost classifier.
        rf_preds: Predictions made by Random Forest on the test set.
        xgb_preds: Predictions made by XGBoost on the test set.
        feature_names: Names of the features used during training.
        y_test: Actual labels of the ground truth testing set.
        y_train: Actual labels of the ground truth training set.
    """

    
    # ---------------------------------------------
    # 1. Train Random Forest
    # ---------------------------------------------
    print("\nTraining Random Forest Model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print("\n=== Random Forest Results ===")
    print(f"Accuracy: {rf_acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, rf_preds, zero_division=0))
    
    # Save the Random Forest model
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Random Forest model saved as 'rf_model.pkl'")
    
    # Print Top 15 Feature Importances for Random Forest
    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    top_15_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 Most Important Features (Random Forest):")
    print(top_15_features)
    
    # ---------------------------------------------
    # 2. Train XGBoost
    # ---------------------------------------------
    print("\nTraining XGBoost Model...")
    # XGBoost >= 1.3.0 requires integer target labels natively, so we encode them
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Identify unique classes and their encoding mappings mapping
    classes = le.classes_
    
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=20, 
        random_state=42, 
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train_encoded)
    
    xgb_preds_encoded = xgb_model.predict(X_test)
    
    # Inverse transform predictions back to string labels for metrics
    xgb_preds = le.inverse_transform(xgb_preds_encoded)
    
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    print("\n=== XGBoost Results ===")
    print(f"Accuracy: {xgb_acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds, zero_division=0))

    return rf_model, xgb_model, rf_preds, xgb_preds, X_train.columns, y_test, y_train

if __name__ == "__main__":
    train_and_evaluate()
