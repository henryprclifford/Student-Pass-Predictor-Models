import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if not os.path.exists("figures"):
    os.mkdir("figures")
if not os.path.exists("figures/model_evaluation"):
    os.mkdir("figures/model_evaluation")

X_train = np.load("model_data/X_train.npy")
X_test = np.load("model_data/X_test.npy") 
y_train = np.load("model_data/y_train.npy")
y_test = np.load("model_data/y_test.npy")
preprocessor = joblib.load("model_data/preprocessor_pipeline.joblib")

feature_names = []
try:
    with open("model_data/feature_names.txt", 'r') as f:
        for line in f:
            if ':' in line:
                feature_names.append(line.split(':', 1)[1].strip())
            else:
                feature_names.append(line.strip())
except Exception as e:
    print(f"Warning: Could not load feature names. Error: {e}")

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Part 2: Model Training with Grid Search
logreg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10, 100],
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

grid_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5, scoring='accuracy')
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')

grid_logreg.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
grid_svm.fit(X_train, y_train)

print("Logistic Regression optimal parameters:", grid_logreg.best_params_)
print("Random Forest optimal parameters:", grid_rf.best_params_)
print("SVM optimal parameters:", grid_svm.best_params_)

knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  
}

grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)

print("KNN optimal parameters:", grid_knn.best_params_)

models = {
    'Logistic Regression': grid_logreg,
    'Random Forest': grid_rf,
    'SVM': grid_svm,
    'KNN': grid_knn
}

for name, model in models.items():
    joblib.dump(model, f"model_data/{name.replace(' ', '_').lower()}_best_model.joblib")

# Part 3: Model Evaluation and Visualization
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 40)

for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    plt.savefig(f"figures/model_evaluation/{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves For Each Model')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/model_evaluation/roc_curves_comparison.png")
plt.close()

metrics = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

metrics_df = pd.DataFrame(metrics).T
print("\nPerformance metrics summary:")
print(metrics_df)

metrics_df.to_csv("model_data/model_performance_metrics.csv")

plt.figure(figsize=(12, 8))
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Comparison of Model Performance Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim([0, 1])
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/model_evaluation/model_performance_comparison.png")
plt.close()

if feature_names and 'Random Forest' in models:
    rf_model = models['Random Forest'].best_estimator_
    importances = rf_model.feature_importances_
    
    if len(feature_names) == len(importances):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        feature_importance_df.to_csv("model_data/feature_importance.csv", index=False)
        
        top_n = min(15, len(feature_names))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
        plt.title(f'Top {top_n} Feature Importance from Random Forest')
        plt.tight_layout()
        plt.savefig("figures/model_evaluation/rf_feature_importance.png")
        plt.close()
        
        print(f"Top {top_n} important features:")
        for i, row in feature_importance_df.head(top_n).iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")

print("\nBest model performance: {0} (Accuracy: {1:.4f})".format(
    metrics_df['Accuracy'].idxmax(), metrics_df['Accuracy'].max()))