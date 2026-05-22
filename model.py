import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/sales.csv")

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Feature Engineering
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

# Convert Category to numerical (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Category'], drop_first=True)

# ✅ FINAL FEATURES (NO DATA LEAKAGE)
X = df.drop(['Sales', 'Order Date', 'Profit'], axis=1)
y = df['Sales']

# ✅ Better split (reduces overfitting)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=None
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
