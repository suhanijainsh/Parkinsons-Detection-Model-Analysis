Parkinson's Disease Prediction - Machine Learning Project
This repository contains the code for a machine learning project that investigates the use of various classification algorithms to predict Parkinson's disease based on voice recordings. The project focuses on handling class imbalance, feature analysis, model selection, and hyperparameter tuning.

Key functionalities:

Data Preprocessing:
Loads and explores the Parkinson's disease dataset.
Handles class imbalance using SMOTE oversampling.
Performs data cleaning and feature engineering.
Standardizes features using StandardScaler.
Exploratory Data Analysis (EDA):
Analyzes the distribution of features for each class.
Creates visualizations like boxplots, violin plots, and correlation heatmaps.
Model Selection and Training:
Implements various classification algorithms including Logistic Regression, Naive Bayes, SVM, Neural Networks, Decision Tree, Random Forest, Gradient Boosting, and KNN.
Evaluates model performance using metrics like accuracy, Matthews correlation coefficient (MCC), and ROC curves.
Performs hyperparameter tuning using GridSearchCV for Gradient Boosting and Random Forest.
Final Model:
Identifies the best performing model based on evaluation metrics.
Defines a function to predict Parkinson's disease for new data points.
Additional Analysis:
Calculates precision, recall, and F1-score for all models.
Provides a comprehensive overview of the final model's performance.
Getting Started:

Clone this repository.
Install the required libraries using pip install -r requirements.txt.
Run the Jupyter Notebook (ML_projectpractice_Final.ipynb) to execute the code and view the results.
Note:

This project is for educational purposes only and should not be used for medical diagnosis.
