# Student-Pass-Predictor-Models
Testing three models on a dataset of various variables. Some are useless, some are useful. When a students grade is equal to or greater than 12 they pass. Each model should be able to predict whether a student will pass based on the other attributes.

Student Performance Analysis and Prediction
Project Overview
This project explores student performance using a dataset containing academic and socio-demographic attributes. The primary objective is to analyse factors influencing student success and build predictive models to classify students as either passing or failing. The project applies statistical analysis, visualisation techniques, and machine learning models to extract insights and evaluate predictive performance.

Data Analysis and Feature Engineering
The dataset includes a range of numerical and categorical features, such as grades, parental education, study habits, and lifestyle factors. Before model training, data preprocessing and feature selection were performed to enhance predictive accuracy.

Key steps included:

Identifying missing values and addressing data inconsistencies.
Statistical hypothesis testing to assess the significance of each feature:
T-tests for numerical features to compare mean differences between passing and failing students.
Chi-square tests for categorical features to examine relationships with student outcomes.
Feature transformations and engineering, including:
Creating new variables such as parental education level (product of maternal and paternal education).
Constructing study time vs. failures ratio to capture the balance between effort and academic struggles.
Averaging weekday and weekend alcohol consumption into a single feature.
One-hot encoding of categorical variables for machine learning compatibility.
These steps refined the dataset, improving the interpretability and performance of the predictive models.

Visualisation and Exploratory Analysis
To understand the dataset better, a series of visualisations were generated:

Pass/Fail Distribution – A bar chart illustrating the overall proportion of passing and failing students.
Feature Impact Analysis – Histogram overlays showing the distribution of key numerical features among pass/fail groups.
Category-Based Pass Rates – Bar plots displaying pass rates across different categorical features (e.g., study time groups, parental education levels).
Feature Importance from Random Forest – A ranked list of the most influential features in classification.
Correlation Heatmaps – Exploring relationships between numerical variables.
These visualisations provided a strong foundation for understanding which attributes most strongly correlate with academic success.

Machine Learning Models and Performance
Several classification models were trained and evaluated to predict student outcomes. Each model was optimised using GridSearchCV with cross-validation to identify the best hyperparameters.

1. Logistic Regression
A simple yet interpretable model, Logistic Regression serves as a baseline.

Strengths: Fast training, interpretable coefficients.
Weaknesses: Limited ability to capture non-linear relationships.
Accuracy: ~79%
2. Random Forest Classifier
An ensemble model that builds multiple decision trees and averages their outputs.

Strengths: Handles non-linearity, robust to outliers, feature importance ranking.
Weaknesses: Computationally expensive, prone to overfitting with too many trees.
Accuracy: ~85% (Best performer)
Top Features:
Study time
Failures
Parental education
Alcohol consumption
3. Support Vector Machine (SVM)
A model that finds the optimal hyperplane for classification.

Strengths: Effective in high-dimensional spaces.
Weaknesses: Slow with large datasets, sensitive to parameter tuning.
Accuracy: ~81%
4. K-Nearest Neighbours (KNN)
A distance-based model that classifies students based on the majority vote of their closest neighbours.

Strengths: Simple, non-parametric, works well with well-separated classes.
Weaknesses: Computationally expensive, struggles with high-dimensional data.
Accuracy: ~78%
Each model was evaluated on test data, with key metrics computed:

Model	Accuracy	Precision	Recall	F1 Score	AUC
Random Forest	85%	0.86	0.84	0.85	0.91
SVM	81%	0.82	0.78	0.80	0.87
Logistic Regression	79%	0.80	0.76	0.78	0.85
KNN	78%	0.79	0.74	0.76	0.83
Random Forest emerged as the most accurate model, demonstrating strong classification performance.

Model Evaluation and Interpretation
Confusion Matrices
Each model's confusion matrix was plotted to illustrate classification errors.

Random Forest exhibited the best balance between false positives and false negatives.
SVM and Logistic Regression misclassified some failing students as passing, highlighting the challenge of boundary cases.
ROC Curves and AUC Scores
Receiver Operating Characteristic (ROC) curves were plotted for all models, with Random Forest achieving the highest AUC of 0.91, indicating superior ability to distinguish between passing and failing students.

Feature Importance
From the Random Forest feature rankings, the most influential predictors were:

Study time – More time dedicated to study directly correlated with passing.
Failures – Past academic failures strongly predicted future failure.
Parental education – Higher parental education levels were linked to better student outcomes.
Alcohol consumption – Higher consumption was negatively correlated with passing rates.
This analysis provides actionable insights into student success factors, reinforcing the importance of study habits and support systems.

Key Findings and Insights
Parental education and study time are the strongest predictors of success.
Alcohol consumption and past failures significantly impact academic performance.
Random Forest outperforms all other models, offering the best predictive accuracy.
Some features (e.g., school type, travel time) had minimal impact and could be dropped in future iterations.
These findings align with common educational research, supporting the idea that structured study habits and a stable home environment contribute to academic success.

Conclusion
This project successfully analysed and predicted student performance using a range of machine learning models. By combining statistical analysis, feature engineering, and predictive modelling, it provides valuable insights into the key factors influencing academic success.

The Random Forest model demonstrated the best classification accuracy and interpretability, making it a strong choice for real-world applications in educational data science.
