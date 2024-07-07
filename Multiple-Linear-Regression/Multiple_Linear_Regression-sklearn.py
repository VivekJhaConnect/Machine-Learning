import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_csv('data.csv')

# Define dependent and independent variables
X = data[['X1', 'X2', 'X3']]  # independent variables
Y = data['Y']  # dependent variable

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the MLR model
model = sm.OLS(Y, X).fit()

# Summary of the model
print(model.summary())
