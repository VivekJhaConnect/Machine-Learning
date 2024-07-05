# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 3, 5, 7, 11])

# Number of observations
n = np.size(X)

# Mean of X and Y
m_X = np.mean(X)
m_Y = np.mean(Y)

# Calculating the cross-deviation and deviation about X
SS_xy = np.sum(Y*X) - n*m_Y*m_X
SS_xx = np.sum(X*X) - n*m_X*m_X

# Calculating regression coefficients
slope = SS_xy / SS_xx
intercept = m_Y - slope*m_X

# Predict values
Y_pred = intercept + slope*X

# Plot the data and the regression line
plt.scatter(X, Y, color='blue', label='Original data')
plt.plot(X, Y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the coefficients
print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
