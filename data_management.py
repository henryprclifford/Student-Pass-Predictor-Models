import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
import joblib

if not os.path.exists("figures"):
    os.mkdir("figures")
if not os.path.exists("model_data"):
    os.mkdir("model_data")  
if not os.path.exists("figures/target_analysis"):
    os.mkdir("figures/target_analysis")
if not os.path.exists("figures/feature_importance"):
    os.mkdir("figures/feature_importance")
if not os.path.exists("figures/feature_relationships"):
    os.mkdir("figures/feature_relationships")

grades_file = "grades_data.xlsx"
data = pd.read_excel(grades_file)

print("\nFirst few rows of data:")
print(data.head(3))

passing_grade_threshold = 12
data['Pass'] = 0
data.loc[data['Grade'] >= passing_grade_threshold, 'Pass'] = 1

print(f"\nShape of dataset: {data.shape[0]} students, {data.shape[1]} features")
missing = data.isnull().sum()
if missing.sum() > 0:
    print("\nWarning: Missing values detected!")
    print(missing[missing > 0])
else:
    print("\nNo missing values - that's good!")

num_cols = []
cat_cols = []

for col in data.columns:
    if col == 'Pass' or col == 'Grade':
        continue
    
    if data[col].dtype == 'object':
        cat_cols.append(col)
    elif data[col].dtype == 'int64' or data[col].dtype == 'float64':
        num_cols.append(col)

print(f"\nFound {len(num_cols)} numerical features and {len(cat_cols)} categorical features")

plt.figure(figsize=(9, 5))
colors = ['#FF5733', '#33FF57']

fail_count = data[data['Pass'] == 0].shape[0]
pass_count = data[data['Pass'] == 1].shape[0]

plt.bar([0, 1], [fail_count, pass_count], color=colors)
plt.title('Pass/Fail Distribution', fontsize=14)
plt.xlabel('Pass Status')
plt.ylabel('Number of Students')
plt.xticks([0, 1], ['Failed', 'Passed'])

plt.text(0, fail_count/2, f"{fail_count}", ha='center')
plt.text(1, pass_count/2, f"{pass_count}", ha='center')

plt.savefig("figures/target_analysis/pass_distribution.png")
plt.close()

pass_rate = (pass_count / (fail_count + pass_count)) * 100
print(f"\nPass rate: {pass_rate:.2f}%")

def my_ttest(data, feature, verbose=True):
    passers = data.loc[data['Pass'] == 1, feature]
    failers = data.loc[data['Pass'] == 0, feature]
    
    t, p = stats.ttest_ind(passers, failers, equal_var=False)
    
    abs_t = abs(t)
    
    if verbose:
        if p < 0.05:
            print(f"{feature}: t={abs_t:.4f}, p={p:.4f} (SIGNIFICANT!)")
        else:
            print(f"{feature}: t={abs_t:.4f}, p={p:.4f}")
            
    return abs_t, p

numeric_importances = {}
print("\n--- NUMERICAL FEATURES VS PASS ---")

plt.figure(figsize=(16, 10))
rows_needed = (len(num_cols) + 3) // 4

i = 1
for feature in num_cols:
    t_val, p_val = my_ttest(data, feature)
    numeric_importances[feature] = t_val
    
    plt.subplot(rows_needed, 4, i)
    i += 1
    
    sns.histplot(data=data[data['Pass']==0], x=feature, color='red', alpha=0.5, label='Fail')
    sns.histplot(data=data[data['Pass']==1], x=feature, color='green', alpha=0.5, label='Pass')
    
    plt.title(f"{feature} (t={t_val:.2f}{'*' if p_val<0.05 else ''})")
    
    if i == 2:
        plt.legend()

plt.tight_layout()
plt.savefig("figures/feature_importance/numeric_features.png")
plt.close()

sorted_num_features = []
for feature in num_cols:
    sorted_num_features.append((feature, numeric_importances[feature]))

sorted_num_features.sort(key=lambda x: x[1], reverse=True)

print("\n--- CATEGORICAL FEATURES VS PASS ---")
cat_importances = {}

for feature in cat_cols:
    crosstab = pd.crosstab(data[feature], data['Pass'])
    
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    
    cat_importances[feature] = chi2
    
    sig_symbol = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{feature}: chi2={chi2:.4f}, p={p:.4f} {sig_symbol}")
    
    plt.figure(figsize=(8, 4))
    
    categories = data[feature].unique()
    pass_rates = []
    
    for category in categories:
        category_data = data[data[feature] == category]
        category_pass_rate = category_data['Pass'].mean() * 100
        pass_rates.append((category, category_pass_rate))
    
    pass_rates.sort(key=lambda x: x[1], reverse=True)
    
    sorted_categories = [x[0] for x in pass_rates]
    sorted_rates = [x[1] for x in pass_rates]
    
    plt.bar(range(len(sorted_categories)), sorted_rates, color='skyblue')
    plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=45)
    plt.title(f'Pass Rate by {feature} (chi2={chi2:.2f}{sig_symbol})')
    plt.ylabel('Pass Rate (%)')
    
    plt.axhline(y=pass_rate, color='red', linestyle='--', label=f'Average: {pass_rate:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/feature_importance/{feature}_pass_rate.png")
    plt.close()

sorted_cat_features = []
for feature in cat_cols:
    sorted_cat_features.append((feature, cat_importances[feature]))

sorted_cat_features.sort(key=lambda x: x[1], reverse=True)

print("\n--- TOP FEATURES FOR PREDICTING PASS ---")
print("Top Numerical Features:")
for i, (feature, importance) in enumerate(sorted_num_features[:5]):
    print(f"{i+1}. {feature:<10} : {importance:.4f}")

print("\nTop Categorical Features:")
for i, (feature, importance) in enumerate(sorted_cat_features[:5]):
    print(f"{i+1}. {feature:<10} : {importance:.4f}")

selected_num_features = []
selected_cat_features = []

for feature, _ in sorted_num_features:
    _, p_val = my_ttest(data, feature, verbose=False)
    if p_val < 0.05:
        selected_num_features.append(feature)

for feature in cat_cols:
    crosstab = pd.crosstab(data[feature], data['Pass'])
    _, p, _, _ = stats.chi2_contingency(crosstab)
    if p < 0.05:
        selected_cat_features.append(feature)

print("\n--- SELECTED FEATURES ---")
print(f"Selected numerical features: {selected_num_features}")
print(f"Selected categorical features: {selected_cat_features}")

data_copy = data.copy()

if 'Medu' in data.columns and 'Fedu' in data.columns:
    data_copy['parent_education'] = data_copy['Medu'] * data_copy['Fedu']
    selected_num_features.append('parent_education')
    print("Added feature: parent_education")

if ('studytime' in data_copy.columns) & ('failures' in data_copy.columns):
    data_copy['study_vs_fail'] = data_copy['studytime'] / (data_copy['failures'] + 1)
    selected_num_features.append('study_vs_fail')
    print("Added feature: study_vs_fail")

try:
    if 'Dalc' in data_copy.columns and 'Walc' in data_copy.columns:
        data_copy['alcohol_avg'] = (data_copy['Dalc'] + data_copy['Walc']) / 2
        selected_num_features.append('alcohol_avg')
        print("Added feature: alcohol_avg")
except:
    print("Couldn't create alcohol_avg feature")

X_with_features = data_copy[selected_num_features + selected_cat_features]
y_target = data_copy['Pass']

my_test_size = 0.2
my_random = 42

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    X_with_features, y_target, 
    test_size=my_test_size, 
    random_state=my_random, 
    stratify=y_target
)

print(f"\nSplit data into training ({X_train_data.shape[0]} samples) and testing ({X_test_data.shape[0]} samples)")

num_transformer = Pipeline([
    ('standardize', StandardScaler())
])

cat_transformer = Pipeline([
    ('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

col_transformer = ColumnTransformer(
    transformers=[
        ('num_features', num_transformer, selected_num_features),
        ('cat_features', cat_transformer, selected_cat_features)
    ]
)

preprocess_pipeline = Pipeline([
    ('transform_cols', col_transformer)
])

print("\nFitting preprocessing pipeline...")
preprocess_pipeline.fit(X_train_data)
print("Pipeline fitted")

print("Transforming train and test data...")
X_train_transformed = preprocess_pipeline.transform(X_train_data)
X_test_transformed = preprocess_pipeline.transform(X_test_data)
print("Data transformed")

all_feature_names = []
all_feature_names.extend(selected_num_features)

if len(selected_cat_features) > 0:
    try:
        cat_encoder = preprocess_pipeline.named_steps['transform_cols'].transformers_[1][1].named_steps['one_hot']
        encoded_feature_names = cat_encoder.get_feature_names_out(selected_cat_features)
        all_feature_names.extend(encoded_feature_names)
    except Exception as e:
        print(f"Warning: Could not extract categorical feature names. Error: {e}")
        print("Using placeholder names for categorical features")
        for cat in selected_cat_features:
            all_feature_names.append(f"{cat}_encoded")

print("\n--- PROCESSED DATASET INFO ---")
print(f"Training data: {X_train_transformed.shape}")
print(f"Testing data: {X_test_transformed.shape}")
print(f"Total features after encoding: {len(all_feature_names)}")
print(f"Class distribution in training: {y_train_data.value_counts().to_dict()}")
print(f"Class distribution in testing: {y_test_data.value_counts().to_dict()}")

output_dir = "model_data"

print("\nSaving processed data to files...")
joblib.dump(preprocess_pipeline, f"{output_dir}/preprocessor_pipeline.joblib")
np.save(f"{output_dir}/X_train.npy", X_train_transformed)
np.save(f"{output_dir}/X_test.npy", X_test_transformed)
np.save(f"{output_dir}/y_train.npy", y_train_data.values)
np.save(f"{output_dir}/y_test.npy", y_test_data.values)

with open(f"{output_dir}/feature_names.txt", 'w') as name_file:
    for i, feature in enumerate(all_feature_names):
        name_file.write(f"{i+1}: {feature}\n")

cv_folds = 5
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=my_random)

cv_params = {
    "cv_type": "StratifiedKFold",
    "n_splits": cv_folds,
    "random_state": my_random,
    "test_size": my_test_size,
    "stratify": "yes"
}

with open(f"{output_dir}/cv_info.txt", 'w') as cv_file:
    cv_file.write("# Cross-validation configuration\n")
    
    for key, value in cv_params.items():
        cv_file.write(f"{key} = {value}\n")

print(f"\nFiles saved to {output_dir}/")
print("\n--- READY FOR MODEL TRAINING ---")