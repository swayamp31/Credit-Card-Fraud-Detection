# Credit-Card-Fraud-Detection
Credit Card Fraud Detection: A Cost-Sensitive Machine Learning Approach

Executive Summary

This project tackles the highly imbalanced nature of financial fraud detection, where fraudulent transactions account for just 0.172% of all data. Rather than relying on standard accuracy, this project optimizes for Precision-Recall Area Under Curve (PR-AUC) and translates model performance into direct Expected Value (Financial ROI).

By comparing a baseline Logistic Regression model against an optimized XGBoost Classifier, the final model achieved an 86% Recall and 72% Precision. More importantly, the XGBoost model prevented an estimated $11,310 in operational losses caused by false positives, generating a positive net business impact of $7,905 on the test set.

The Business Problem

In fraud detection, institutions face a strict trade-off:

False Negatives (Missed Fraud): Results in direct financial loss and chargebacks.

False Positives (False Alarms): Results in frozen accounts, operational investigation costs, and severe customer dissatisfaction.

Goal: Build a model that maximizes fraud detection (Recall) while maintaining high enough Precision to keep the operational costs of false alarms lower than the money saved.

Tech Stack & Methods

Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Matplotlib, Seaborn

Techniques Used:

Robust Scaling (outlier mitigation)

Stratified Train/Test Splitting (preventing data leakage)

SMOTE (Synthetic Minority Over-sampling Technique)

Cost-Benefit Matrix Modeling

Key Findings & Results

The Accuracy Paradox: A naive model predicting "No Fraud" on this dataset achieves 99.8% accuracy but is financially useless.

Baseline vs. Optimized: The baseline Logistic Regression caught 92% of fraud but flagged 1,410 innocent transactions, resulting in a net financial loss. The XGBoost model caught 86% of fraud but reduced false alarms to just 33, proving vastly superior for business deployment.

Financial Impact: The XGBoost model delivered a $19,215 value add over the baseline model based on our simulated cost-benefit matrix.

How to View & Run the Project

This project was developed and executed entirely within Kaggle, meaning no local environment setup or massive data downloads are required!

View the Analysis: Read through the complete, interactive notebook here: https://www.kaggle.com/code/swaggpale/creditcard-fraud-analysis

Run the Code: The creditcard.csv dataset is automatically attached to the Kaggle environment. To run or modify the code yourself, simply click "Copy & Edit" in the top right corner of the Kaggle notebook to fork the environment and execute the cells.

Created as a comprehensive portfolio project demonstrating end-to-end data processing, machine learning, and financial impact analysis.
