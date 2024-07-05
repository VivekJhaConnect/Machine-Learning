# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
Y = np.array([2, 3, 5, 7, 11])  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, Y, color='blue', label='Original data')
plt.plot(X, Y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")
