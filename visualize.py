import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, f1_score
)


def create_visualizations(rf_model, xgb_model, rf_preds, xgb_preds,
                          feature_names, y_test, y_train, dataset="nslkdd"):
    """
    Generates and saves 5 evaluation plots to results/.

    Args:
        dataset: "nslkdd" or "unsw"
                 Controls output filenames and plot titles.
    """
    print("\nGenerating Visualizations...")
    os.makedirs('results', exist_ok=True)

    # File prefix and display name per dataset
    if dataset == "unsw":
        prefix = "unsw_"
        dname  = "UNSW-NB15"
    else:
        prefix = ""
        dname  = "NSL-KDD"

    labels = sorted(y_test.unique())

    # ------------------------------------------------------------------
    # 1. Attack Category Distribution
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    counts = y_train.value_counts()
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = plt.bar(counts.index, counts.values, color=colors)
    plt.title(f"Attack Category Distribution — {dname} (Training Data)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Attack Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + yval * 0.01,
                 int(yval), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    path = os.path.abspath(f'results/{prefix}attack_category_distribution.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # 2. Confusion Matrix (Random Forest)
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, rf_preds, labels=labels)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc, 0)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            p = cm_perc[i, j]
            annot[i, j] = f"{cm[i, j]}\n({p:.1%})"

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {dname} (Random Forest)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.tight_layout()
    path = os.path.abspath(f'results/{prefix}confusion_matrix.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # 3. Feature Importance (Random Forest, top 15)
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_15 = importances.sort_values(ascending=True).tail(15)
    sns.barplot(x=top_15.values, y=top_15.index,
                hue=top_15.values, palette='viridis', legend=False)
    plt.title(f"Top 15 Feature Importances — {dname} (Random Forest)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    path = os.path.abspath(f'results/{prefix}feature_importance.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # 4. Model Comparison (RF vs XGBoost, weighted metrics)
    # ------------------------------------------------------------------
    rf_acc  = accuracy_score(y_test, rf_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    rf_p,  rf_r,  rf_f,  _ = precision_recall_fscore_support(
        y_test, rf_preds,  average='weighted', zero_division=0)
    xgb_p, xgb_r, xgb_f, _ = precision_recall_fscore_support(
        y_test, xgb_preds, average='weighted', zero_division=0)

    # Also report macro F1 as subtitle annotation
    rf_macro  = f1_score(y_test, rf_preds,  average='macro', zero_division=0)
    xgb_macro = f1_score(y_test, xgb_preds, average='macro', zero_division=0)

    metrics    = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1 (W)']
    rf_scores  = [rf_acc,  rf_p,  rf_r,  rf_f]
    xgb_scores = [xgb_acc, xgb_p, xgb_r, xgb_f]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, rf_scores,  width,
                   label=f'Random Forest (Macro F1={rf_macro:.2f})',  color='steelblue')
    bars2 = ax.bar(x + width / 2, xgb_scores, width,
                   label=f'XGBoost (Macro F1={xgb_macro:.2f})', color='darkorange')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison — {dname}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])

    for b in list(bars1) + list(bars2):
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2., yval + 0.01,
                f"{yval:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    path = os.path.abspath(f'results/{prefix}model_comparison.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # 5. Per-Class Performance (Random Forest)
    # ------------------------------------------------------------------
    report = classification_report(y_test, rf_preds,
                                   output_dict=True, zero_division=0)
    classes  = [c for c in labels if c in report]
    p_scores  = [report[c]['precision'] for c in classes]
    r_scores  = [report[c]['recall']    for c in classes]
    f1_scores = [report[c]['f1-score']  for c in classes]

    x_class   = np.arange(len(classes))
    bar_width  = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_class - bar_width, p_scores,  bar_width,
           label='Precision', color='lightcoral')
    ax.bar(x_class,             r_scores,  bar_width,
           label='Recall',    color='mediumseagreen')
    ax.bar(x_class + bar_width, f1_scores, bar_width,
           label='F1 Score',  color='cornflowerblue')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance — {dname} (Random Forest)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_class)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    plt.tight_layout()
    path = os.path.abspath(f'results/{prefix}per_class_performance.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ------------------------------------------------------------------
# CLI entry points
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate result plots for NSL-KDD or UNSW-NB15."
    )
    parser.add_argument(
        "--dataset", choices=["nslkdd", "unsw"], default="nslkdd",
        help="Which dataset to generate plots for (default: nslkdd)"
    )
    args = parser.parse_args()

    if args.dataset == "nslkdd":
        from train import train_and_evaluate
        from preprocess import load_and_preprocess_data

        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
        rf_model, xgb_model, rf_preds, xgb_preds, fnames, y_test, y_train = \
            train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
        create_visualizations(rf_model, xgb_model, rf_preds, xgb_preds,
                              fnames, y_test, y_train, dataset="nslkdd")

    else:  # unsw
        import joblib
        from preprocess_unsw import load_and_preprocess_unsw
        from imblearn.over_sampling import SMOTE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test, _ = load_and_preprocess_unsw()

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)

        rf_model = joblib.load('rf_model_unsw.pkl')
        rf_preds = rf_model.predict(X_test)

        le = LabelEncoder()
        xgb_model = XGBClassifier(
            n_estimators=100, max_depth=20, random_state=42,
            n_jobs=-1, eval_metric='mlogloss', verbosity=0
        )
        xgb_model.fit(X_bal, le.fit_transform(y_bal))
        xgb_preds = le.inverse_transform(xgb_model.predict(X_test))

        create_visualizations(rf_model, xgb_model, rf_preds, xgb_preds,
                              X_train.columns, y_test, y_train, dataset="unsw")