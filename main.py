import sys
from sklearn.metrics import accuracy_score, f1_score

from preprocess import load_and_preprocess_data
from train import train_and_evaluate
from visualize import create_visualizations


def main():
    print("Starting NIDS Pipeline...")

    try:
        # Step 1: Preprocess Data
        print("\n--- Step 1: Preprocessing Data ---")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

        # Step 2: Train Models
        print("\n--- Step 2: Training Models ---")
        rf_model, xgb_model, rf_preds, xgb_preds, fnames, y_test_out, y_train_out = (
            train_and_evaluate(
                X_train,
                X_test,
                y_train,
                y_test,
                preprocessor
            )
        )

        # Calculate accuracy for summary
        rf_acc = accuracy_score(y_test, rf_preds)
        xgb_acc = accuracy_score(y_test, xgb_preds)

        # Step 3: Visualize Results
        print("\n--- Step 3: Generating Visualizations ---")
        create_visualizations(
            rf_model,
            xgb_model,
            rf_preds,
            xgb_preds,
            fnames,
            y_test_out,
            y_train_out
        )

        # Final Summary
        print("\n==========================================")
        print("   NIDS Results Summary")
        print("==========================================")
        print(f"Training samples: {len(y_train)}")
        print(f"Testing samples: {len(y_test)}")
        print(f"Random Forest  — Macro F1: {f1_score(y_test, rf_preds, average='macro', zero_division=0):.4f}")
        print(f"XGBoost        — Macro F1: {f1_score(y_test, xgb_preds, average='macro', zero_division=0):.4f}")
        print("Results saved to: results/")
        print("==========================================")

    except Exception as e:
        print(f"\n[ERROR] An error occurred in the NIDS Pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()