import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

if not os.path.exists("figures/test_evaluation"):
    os.makedirs("figures/test_evaluation")

X_test = np.load("model_data/X_test.npy")
y_test = np.load("model_data/y_test.npy")

model_files = [f for f in os.listdir("model_data") if f.endswith("_best_model.joblib")]
models = {}

for model_file in model_files:
    name = model_file.replace("_best_model.joblib", "").replace("_", " ").title()
    models[name] = joblib.load(f"model_data/{model_file}")

metrics_dict = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'AUC': []
}

plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics_dict['Model'].append(name)
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['Precision'].append(precision)
    metrics_dict['Recall'].append(recall)
    metrics_dict['F1 Score'].append(f1)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    metrics_dict['AUC'].append(roc_auc)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"figures/test_evaluation/{name.lower().replace(' ', '_')}_cm.png")
    plt.close()
    
    print(f"{name} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {roc_auc:.4f}")
    print("-" * 40)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/test_evaluation/roc_comparison.png")
plt.close()

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("model_data/test_metrics.csv", index=False)

plt.figure(figsize=(12, 8))
model_names = metrics_df['Model']
metrics_plot = metrics_df.drop('Model', axis=1)
metrics_plot.index = model_names

ax = metrics_plot.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.tight_layout()
plt.savefig("figures/test_evaluation/model_metrics_comparison.png")

best_model_name = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
best_accuracy = metrics_df['Accuracy'].max()

print(f"\nBest model: {best_model_name} (Accuracy: {best_accuracy:.4f})")