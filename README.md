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

## Conclusion

The analysis and prediction of stock prices are complex tasks that involve various factors and uncertainties. In this project, we've used machine learning techniques to attempt to predict Tesla stock price movement. The models, including Logistic Regression, Support Vector Classifier, and XGBoost Classifier, were trained and evaluated. The results indicated that the Logistic Regression model performed best among the tested models, but its accuracy was not significantly better than random guessing.

It's important to note that stock price prediction is influenced by many external factors, and using historical data alone might not be sufficient for accurate predictions. This project serves as an example of how to approach stock prediction with machine learning, but further analysis and incorporation of more sophisticated techniques may be needed for more reliable predictions.

Feel free to experiment with different models, features, and parameters to potentially improve the prediction performance.

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
