import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('2025_03_13_genomic_trajectory_SNP_table_withPvalues.csv')

# Convert p-values to -log10(p-value) (Makes it easier to fit)
data['negative_log_pvalue'] = -np.log10(data['pvalue'])

# Prepare features and target
X = data[['convertedPosition', 'negative_log_pvalue']]
y = data.iloc[:, 4:-2].mean(axis=1)  # Average frequency across all samples

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Print model coefficients
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot of the data
plt.scatter(X_test['convertedPosition'], y_test, color='blue', alpha=0.5, label='Actual data')

# Sorting the test data for a smooth line plot
sort_idx = X_test['convertedPosition'].argsort()
plt.plot(X_test['convertedPosition'].iloc[sort_idx], y_pred[sort_idx], color='red', label='Regression line')

plt.xlabel('Converted Position')
plt.ylabel('Average Frequency')
plt.title('Linear Regression: Converted Position vs Average Frequency')
plt.legend()

plt.tight_layout()
plt.show() # The plot is almost linear, I think I might have used the wrong position?
