import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/sales.csv")

print("Dataset Preview:\n", df.head())

# Basic info
print("\nInfo:")
print(df.info())

# Summary
print("\nSummary:")
print(df.describe())

# Convert date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sales by category
plt.figure()
sns.barplot(x='Category', y='Sales', data=df)
plt.title("Sales by Category")
plt.xticks(rotation=45)
plt.savefig("outputs/category_sales.png")

# Monthly sales trend
df['Month'] = df['Order Date'].dt.month

plt.figure()
sns.lineplot(x='Month', y='Sales', data=df)
plt.title("Monthly Sales Trend")
plt.savefig("outputs/monthly_sales.png")

# Correlation
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")

print("✔ EDA Completed")
