import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the CSV file
file_path = 'data/ZM.csv'
ZM = pd.read_csv(file_path)
list(ZM)

ZM.csv.out
file_path = 'data/ZM.csv'
ZM = pd.read_csv(file_path)
list(ZM)

# ZM_csv_out
file_path = 'data/ZM.csv.out'
ZM_csv_out = pd.read_csv(file_path)
list(ZM_csv_out)

identifier_columns= ['BranchId', 'ProductId']
prefixes = {'BranchId': 'B', 'ProductId': 'P'}
for col in ['BranchId', 'ProductId']:
    if col in data.columns:
        data[col] = data[col].astype(str)
    else:
        raise ValueError(f"Identifier column '{col}' is missing in the input data.")
for col in identifier_columns:
    if col in data.columns:
        prefix = prefixes.get(col, '')
        data[col] = prefix + data[col].astype(str)
    else:
        raise ValueError(f"Identifier column '{col}' is missing in the input data.")

data.head().to_csv('data/cf_sampledata_exp.csv', index=False)
data.dtypes
# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(data.head())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data types
print("\nData Types:")
print(data.dtypes)
data = data[~data.CashOutOrdersAmount.isna()]
data.CashOutOrdersAmount.astype(float)
data.loc[:,'CashOutOrdersAmount'] = data['CashOutOrdersAmount'].str.replace(',', '').astype(float)
data.columns[data.dtypes==object]
# Unique values per column
print("\nUnique Values:")
for column in data.columns:
    print(f"{column}: {data[column].nunique()} unique values")

# Pairplot of all numerical features
sns.pairplot(data)
plt.suptitle('Pairplot of the Numerical Features', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Histograms for all numerical columns
data.hist(figsize=(16, 12), bins=20)
plt.suptitle('Histograms of the Numerical Features')
plt.show()

# Boxplots for numerical columns
for column in data.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(y=data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Normality tests
print("\nNormality Tests (Shapiro-Wilk):")
for column in data.select_dtypes(include=np.number).columns:
    stat, p = stats.shapiro(data[column].dropna())
    print(f'{column}: Statistics={stat:.3f}, p={p:.3f}')

# Scatter plots between some selected features
# Modify the following as per your specific columns
features_to_plot = ['Column1', 'Column2', 'Column3']
for i in range(len(features_to_plot)):
    for j in range(i + 1, len(features_to_plot)):
        sns.scatterplot(x=data[features_to_plot[i]], y=data[features_to_plot[j]])
        plt.title(f'Scatter Plot between {features_to_plot[i]} and {features_to_plot[j]}')
        plt.show()


#

import matplotlib.pyplot as plt

# Plot the cash balance data
data.plot(figsize=(12, 6))
plt.title('Daily Cash Balance')
plt.ylabel('Cash Balance')
plt.xlabel('Date')
plt.show()
