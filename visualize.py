import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from train import train_and_evaluate

def create_visualizations(rf_model, xgb_model, rf_preds, xgb_preds, feature_names, y_test, y_train):
    """
    Generates and saves 5 evaluation plots to the results/ folder.
    """
    print("\nGenerating Visualizations...")
    
    # Create results folder
    os.makedirs('results', exist_ok=True)
    
    # 1. Attack Category Distribution
    plt.figure(figsize=(10, 6))
    counts = y_train.value_counts()
    # Using a distinct color palette
    colors = sns.color_palette("Set2", n_colors=len(counts))
    bars = plt.bar(counts.index, counts.values, color=colors)
    
    plt.title("Attack Category Distribution (Training Data)", fontsize=14, fontweight='bold')
    plt.xlabel("Attack Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.01), int(yval), 
                 ha='center', va='bottom', fontsize=10)
                 
    plt.tight_layout()
    dist_path = os.path.abspath('results/attack_category_distribution.png')
    plt.savefig(dist_path, dpi=150)
    print(f"Saved: {dist_path}")
    plt.close()

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    # Extract sorted labels dynamically based on test data classes
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, rf_preds, labels=labels)
    
    # Calculate percentages for annotations
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc, 0) # Handle divide by zero if a class has 0 true instances
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            p = cm_perc[i, j]
            if p > 0:
                annot[i, j] = f"{cm[i, j]}\n({p:.1%})"
            else:
                annot[i, j] = f"{cm[i, j]}\n(0.0%)"
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix Heatmap (Random Forest)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.abspath('results/confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"Saved: {cm_path}")
    plt.close()

    # 3. Feature Importance Chart
    plt.figure(figsize=(10, 8))
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    # Get top 15 and sort ascending so the most important prints at the top in barh
    top_15 = importances.sort_values(ascending=True).tail(15)
    
    # seaborn barplot handles horizontal if x/y are numeric/categorical correctly
    # Color by importance value using hue and palette
    sns.barplot(x=top_15.values, y=top_15.index, hue=top_15.values, palette='viridis', legend=False)
    
    plt.title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    fi_path = os.path.abspath('results/feature_importance.png')
    plt.savefig(fi_path, dpi=150)
    print(f"Saved: {fi_path}")
    plt.close()

    # 4. Model Comparison Chart
    rf_acc = accuracy_score(y_test, rf_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    # Using weighted average for multi-class precision, recall, f1
    rf_p, rf_r, rf_f, _ = precision_recall_fscore_support(y_test, rf_preds, average='weighted', zero_division=0)
    xgb_p, xgb_r, xgb_f, _ = precision_recall_fscore_support(y_test, xgb_preds, average='weighted', zero_division=0)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    rf_scores = [rf_acc, rf_p, rf_r, rf_f]
    xgb_scores = [xgb_acc, xgb_p, xgb_r, xgb_f]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='steelblue')
    bars2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='darkorange')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Comparison: Random Forest vs XGBoost', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15]) # slightly higher to fit labels
    
    # Label bars with values
    for b in bars1:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)
    for b in bars2:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    comp_path = os.path.abspath('results/model_comparison.png')
    plt.savefig(comp_path, dpi=150)
    print(f"Saved: {comp_path}")
    plt.close()

    # 5. Per Class Performance
    report = classification_report(y_test, rf_preds, output_dict=True, zero_division=0)
    
    # Extract only the exact target classes, ignoring 'accuracy', 'macro avg', 'weighted avg'
    classes = [c for c in labels if c in report]
    p_scores = [report[c]['precision'] for c in classes]
    r_scores = [report[c]['recall'] for c in classes]
    f1_scores = [report[c]['f1-score'] for c in classes]
    
    x_class = np.arange(len(classes))
    bar_width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_p = ax.bar(x_class - bar_width, p_scores, bar_width, label='Precision', color='lightcoral')
    bars_r = ax.bar(x_class, r_scores, bar_width, label='Recall', color='mediumseagreen')
    bars_f = ax.bar(x_class + bar_width, f1_scores, bar_width, label='F1 Score', color='cornflowerblue')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Per-Class Performance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_class)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.15])
    
    plt.tight_layout()
    pc_path = os.path.abspath('results/per_class_performance.png')
    plt.savefig(pc_path, dpi=150)
    print(f"Saved: {pc_path}")
    plt.close()

if __name__ == "__main__":
    rf_model, xgb_model, rf_preds, xgb_preds, fnames, y_test, y_train = train_and_evaluate()
    create_visualizations(rf_model, xgb_model, rf_preds, xgb_preds, fnames, y_test, y_train)
