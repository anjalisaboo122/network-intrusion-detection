import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

from preprocess import load_and_preprocess_data

def generate_shap_explanations():
    os.makedirs('results', exist_ok=True)

    # Load saved model and fresh data
    print("Loading model and data...")
    rf_model = joblib.load('rf_model.pkl')
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    # Use a background sample for speed — SHAP TreeExplainer is exact for RF
    # but we limit test samples to 500 so plots are readable
    X_explain = X_test.iloc[:500]
    y_explain = y_test.iloc[:500]

    print("Computing SHAP values (takes ~2 mins)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_explain)

    print(type(shap_values))

    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            print(f"Class {i}: {sv.shape}")
    else:
            print("Shape:", shap_values.shape)

    # shap_values is a list of arrays, one per class
    # rf_model.classes_ gives the class order
    class_names = list(rf_model.classes_)
    print(f"Classes: {class_names}")

    # ------------------------------------------------------------------
    # Plot 1: Global summary — beeswarm for each class
    # Shows which features drive each attack category overall
    # ------------------------------------------------------------------
    print("Saving global SHAP summary plot...")
    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 6))

    for i, cls in enumerate(class_names):
        plt.sca(axes[i])
        shap.summary_plot(
            shap_values[:, :, i],
            X_explain,
            plot_type='dot',
            max_display=10,
            show=False,
            color_bar=(i == len(class_names) - 1)
        )
        axes[i].set_title(f'SHAP — {cls}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/shap_summary.png")

    # ------------------------------------------------------------------
    # Plot 2: Single-sample waterfall — one example per class
    # Shows WHY the model made a specific prediction
    # ------------------------------------------------------------------
    print("Saving per-class single-sample SHAP waterfall plots...")

    for cls in class_names:
        # Find first test sample actually belonging to this class
        matching = y_explain[y_explain == cls]
        if len(matching) == 0:
            print(f"  No {cls} samples in first 500 rows, skipping.")
            continue

        sample_idx = matching.index[0]
        position = X_explain.index.get_loc(sample_idx)
        class_idx = class_names.index(cls)

        shap_exp = shap.Explanation(
            values=shap_values[class_idx][position],
            base_values=explainer.expected_value[class_idx],
            data=X_explain.iloc[position],
            feature_names=list(X_explain.columns)
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, max_display=12, show=False)
        plt.title(f'SHAP Explanation — {cls} sample', fontsize=13, fontweight='bold')
        plt.tight_layout()

        fname = f'results/shap_waterfall_{cls.lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    print("\nSHAP explanations complete.")
    print("Files saved:")
    print("  results/shap_summary.png          <- global, all classes")
    print("  results/shap_waterfall_*.png       <- one per class, local explanation")

if __name__ == "__main__":
    generate_shap_explanations()