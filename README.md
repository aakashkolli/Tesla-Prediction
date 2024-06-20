# Tesla Stock Prediction

## Overview

This Python script aims to predict Tesla stock price movement using machine learning techniques. It performs data analysis, feature engineering, and model development to predict whether the stock price will go up or down.

The script performs the following steps:

1. Imports necessary libraries and modules for data manipulation, visualization, and machine learning.
2. Reads historical Tesla stock data from a CSV file (`TSLA.csv`).
3. Visualizes the data by plotting the closing prices and analyzing data distributions.
4. Removes redundant columns and checks for missing values in the dataset.
5. Extracts features from the date and adds additional features based on the stock data.
6. Normalizes the data and splits it into training and validation sets.
7. Develops and evaluates machine learning models including Logistic Regression, Support Vector Classifier (SVC), and XGBoost Classifier.
8. Prints training and validation accuracies for each model.
9. Displays a pie chart showing the distribution of the target variable.
10. Displays a heatmap to visualize correlations between features.

## Requirements

The following libraries and modules are used in this project:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

You can install these libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
