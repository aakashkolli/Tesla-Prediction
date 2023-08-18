# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
# ML Model Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('TSLA.csv')

# Preview the dataset
# df.head()
# df.shape
# df.describe()
# df.info()

# Data Visualizations
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in USD')
plt.show()

# remove redundant Adjusted Close column
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)

# Check for nulls
df.isnull().sum()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
 
plt.subplots(figsize=(20,10))
 
# Distribution Plots 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()

# Box Plots
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])

# Extract day, month, and year from 'Date' column
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
 
# df.head()

# Add Quarterly feature b/c quarterly results affect stock performance
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
# df.head()

data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

df.groupby('is_quarter_end').mean()

# Add more features
# Target feature (buy/sell signals)
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# check if target is balanced
plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10))
 
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# Split data for training and testing, normalize data
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# Develop and Evaluate Models
models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

# XGBClassifier has overfitting, LogisticRegression is best model, but is no better than 50/50
for i in range(3):
  models[i].fit(X_train, Y_train)
 
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

