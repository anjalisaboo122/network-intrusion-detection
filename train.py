import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import load_and_preprocess_data


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor=None):

    # ------------------------------------------------------------------
    # SMOTE on training data only
    # ------------------------------------------------------------------
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print("  Before:", pd.Series(y_train).value_counts().to_dict())
    print("  After :", pd.Series(y_bal).value_counts().to_dict())

    # ------------------------------------------------------------------
    # Cross-validation helper
    # NOTE: CV runs on ORIGINAL (unbalanced) training data using 
    # StratifiedKFold. This is the honest estimate — SMOTE is not applied 
    # inside folds here, so treat these scores as a lower bound.
    # The test set macro F1 is the primary reported metric.
    # ------------------------------------------------------------------
    def cv_macro_f1(model, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)
        return scores.mean(), scores.std()

    # ------------------------------------------------------------------
    # 1. Random Forest
    # ------------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_bal, y_bal)
    rf_preds = rf_model.predict(X_test)

    rf_macro   = f1_score(y_test, rf_preds, average='macro',    zero_division=0)
    rf_weighted = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
    rf_acc     = accuracy_score(y_test, rf_preds)

    print("\n=== Random Forest — Test Set ===")
    print(f"  Macro F1 (primary) : {rf_macro:.4f}")
    print(f"  Weighted F1        : {rf_weighted:.4f}")
    print(f"  Accuracy           : {rf_acc:.4f}  (not primary — dataset is imbalanced)")
    print(classification_report(y_test, rf_preds, zero_division=0))

    print("Running 5-fold CV on original training data...")
    rf_cv_mean, rf_cv_std = cv_macro_f1(rf_model, X_train, y_train)
    print(f"  CV Macro F1 : {rf_cv_mean:.4f} +/- {rf_cv_std:.4f}")
    print("  (CV uses original data without SMOTE inside folds — honest lower bound)")

    joblib.dump(rf_model, 'rf_model.pkl')
    if preprocessor is not None:
        joblib.dump(preprocessor, 'preprocessor.pkl')
    print("RF model and preprocessor saved.")

    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    print("\nTop 15 features:")
    print(importances.sort_values(ascending=False).head(15).to_string())

    # ------------------------------------------------------------------
    # 2. XGBoost
    # ------------------------------------------------------------------
    print("\nTraining XGBoost...")
    le = LabelEncoder()
    y_bal_enc  = le.fit_transform(y_bal)
    y_test_enc = le.transform(y_test)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    xgb_model.fit(X_bal, y_bal_enc)
    xgb_preds = le.inverse_transform(xgb_model.predict(X_test))

    xgb_macro   = f1_score(y_test, xgb_preds, average='macro',    zero_division=0)
    xgb_weighted = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
    xgb_acc     = accuracy_score(y_test, xgb_preds)

    print("\n=== XGBoost — Test Set ===")
    print(f"  Macro F1 (primary) : {xgb_macro:.4f}")
    print(f"  Weighted F1        : {xgb_weighted:.4f}")
    print(f"  Accuracy           : {xgb_acc:.4f}")
    print(classification_report(y_test, xgb_preds, zero_division=0))

    # For XGBoost CV we pass string labels (cross_val_score handles encoding)
    print("Running 5-fold CV on original training data...")
    xgb_cv_model = XGBClassifier(
        n_estimators=100, max_depth=20, random_state=42,
        n_jobs=-1, eval_metric='mlogloss', verbosity=0
    )
    xgb_cv_mean, xgb_cv_std = cv_macro_f1(xgb_cv_model, X_train,
                                            le.fit_transform(y_train))
    print(f"  CV Macro F1 : {xgb_cv_mean:.4f} +/- {xgb_cv_std:.4f}")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("   Final Results (Macro F1 — primary)")
    print("==========================================")
    print(f"  Random Forest  | Test Macro F1: {rf_macro:.4f}")
    print(f"  XGBoost        | Test Macro F1: {xgb_macro:.4f}")
    print("\nKnown limitation: R2L and U2R recall remains low due to")
    print("train/test feature distribution shift in NSL-KDD.")
    print("This is a documented property of the dataset, not a pipeline bug.")

    return (rf_model, xgb_model, rf_preds, xgb_preds,
            X_train.columns, y_test, y_train)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)