import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='h'),
    'temperature': np.random.normal(loc=70, scale=5, size=100),
    'vibration': np.random.normal(loc=0.5, scale=0.1, size=100),
    'pressure': np.random.normal(loc=30, scale=2, size=100),
    'failure': np.random.choice([0, 1], size=100)
}
df = pd.DataFrame(data)


df.set_index('timestamp', inplace=True)

# Feature Engineering 

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['rolling_mean_temp'] = df['temperature'].rolling(window=3).mean()
df['rolling_std_temp'] = df['temperature'].rolling(window=3).std()
df['rolling_mean_vibration'] = df['vibration'].rolling(window=3).mean()
df['rolling_std_vibration'] = df['vibration'].rolling(window=3).std()
df['rolling_mean_pressure'] = df['pressure'].rolling(window=3).mean()
df['rolling_std_pressure'] = df['pressure'].rolling(window=3).std()
df['lag_temp_1'] = df['temperature'].shift(1)
df['lag_vibration_1'] = df['vibration'].shift(1)
df['lag_pressure_1'] = df['pressure'].shift(1)
df['cumulative_failures'] = df['failure'].cumsum()
df['temp_vibration_interaction'] = df['temperature'] * df['vibration']
df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
df['ema_temp'] = df['temperature'].ewm(span=5, adjust=False).mean()
df['ema_vibration'] = df['vibration'].ewm(span=5, adjust=False).mean()
df['ema_pressure'] = df['pressure'].ewm(span=5, adjust=False).mean()

# Drop rows with NaN values 
df.dropna(inplace=True)

# Prepare the data for modeling
X = df.drop(columns=['failure'])  # Features
y = df['failure']                  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization of Predicted vs Actual Failures
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Failures', color='red', marker='o', linestyle='None')
plt.plot(y_pred, label='Predicted Failures', color='green', marker='x', linestyle='None')
plt.title('Predicted vs Actual Failures')
plt.xlabel('Sample Index')
plt.ylabel('Failure (0 or 1)')
plt.xticks(ticks=np.arange(len(y_test)), labels=np.arange(len(y_test)))
plt.legend()
plt.grid()
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()