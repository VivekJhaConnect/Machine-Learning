import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Generate some random data for the purpose of this example
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 100)
y = x - 2 * (x ** 2) + np.random.normal(-3, 3, 100)

x = x[:, np.newaxis]
y = y[:, np.newaxis]

# Create Polynomial Features
def create_polynomial_features(x, degree):
    x_poly = np.ones((x.shape[0], 1))  # Start with the bias (intercept) term
    for i in range(1, degree + 1):
        x_poly = np.hstack((x_poly, x ** i))
    return x_poly

degree = 2
x_poly = create_polynomial_features(x, degree)

# Compute the Coefficients Using the Normal Equation
def compute_coefficients(x_poly, y):
    # Normal Equation: (X^T * X)^-1 * X^T * y
    x_transpose = x_poly.T
    theta = np.linalg.inv(x_transpose.dot(x_poly)).dot(x_transpose).dot(y)
    return theta

theta = compute_coefficients(x_poly, y)

# Make Predictions
def predict(x_poly, theta):
    return x_poly.dot(theta)

y_pred = predict(x_poly, theta)

# Evaluate the Model
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Visualize the Results
plt.scatter(x, y, s=10)
# Sort the values of x before line plot
sort_axis = np.argsort(x[:, 0])
x_sorted = x[sort_axis]
y_pred_sorted = y_pred[sort_axis]
plt.plot(x_sorted, y_pred_sorted, color='m')
plt.show()
