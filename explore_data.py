import pandas as pd

df = pd.read_csv('customer_data.csv')

print("=== First 5 rows ===")
print(df.head())

print("\n=== Shape (rows, columns) ===")
print(df.shape)

print("\n=== Column types ===")
print(df.dtypes)

print("\n=== Basic stats ===")
print(df.describe())

print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== Churn breakdown ===")
print(df['churned'].value_counts())