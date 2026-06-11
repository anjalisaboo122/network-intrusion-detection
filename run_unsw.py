import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess_unsw import load_and_preprocess_unsw


def run_unsw_pipeline():
    print("=" * 50)
    print("UNSW-NB15 Intrusion Detection Pipeline")
    print("=" * 50)

    # Load and preprocess
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_unsw()

    # SMOTE on training data only
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print("  Before:", pd.Series(y_train).value_counts().to_dict())
    print("  After :", pd.Series(y_bal).value_counts().to_dict())

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_bal, y_bal)
    rf_preds = rf.predict(X_test)

    rf_macro = f1_score(y_test, rf_preds, average='macro', zero_division=0)
    print(f"\n=== Random Forest — UNSW-NB15 Test Set ===")
    print(f"  Macro F1: {rf_macro:.4f}")
    print(classification_report(y_test, rf_preds, zero_division=0))

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    print("\nTraining XGBoost...")
    le = LabelEncoder()
    y_bal_enc  = le.fit_transform(y_bal)
    y_test_enc = le.transform(y_test)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    xgb.fit(X_bal, y_bal_enc)
    xgb_preds = le.inverse_transform(xgb.predict(X_test))

    xgb_macro = f1_score(y_test, xgb_preds, average='macro', zero_division=0)
    print(f"\n=== XGBoost — UNSW-NB15 Test Set ===")
    print(f"  Macro F1: {xgb_macro:.4f}")
    print(classification_report(y_test, xgb_preds, zero_division=0))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("  UNSW-NB15 Final Results")
    print("==========================================")
    print(f"  Random Forest  | Macro F1: {rf_macro:.4f}")
    print(f"  XGBoost        | Macro F1: {xgb_macro:.4f}")
    print("\nComparison with NSL-KDD:")
    print("  NSL-KDD  RF  Macro F1: 0.4855")
    print("  NSL-KDD  XGB Macro F1: 0.5728")
    print(f"  UNSW-NB15 RF  Macro F1: {rf_macro:.4f}")
    print(f"  UNSW-NB15 XGB Macro F1: {xgb_macro:.4f}")

    # Save UNSW models separately
    joblib.dump(rf,           'rf_model_unsw.pkl')
    joblib.dump(preprocessor, 'preprocessor_unsw.pkl')
    print("\nSaved: rf_model_unsw.pkl, preprocessor_unsw.pkl")


if __name__ == "__main__":
    run_unsw_pipeline()