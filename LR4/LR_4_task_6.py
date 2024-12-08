import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

plt.scatter(X, y, color='blue')
plt.plot(X, y_poly_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()

print('Intercept:', poly_reg.intercept_)
print('Coefficients:', poly_reg.coef_)

mse_linear = mean_squared_error(y, y_pred)
mse_poly = mean_squared_error(y, y_poly_pred)

print('MSE Linear Regression:', mse_linear)
print('MSE Polynomial Regression:', mse_poly)



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
lin_reg = LinearRegression()

# Initialize lists to store the training and validation errors
train_errors = []
val_errors = []

# Iterate over different training set sizes
for i in range(10, 61, 10):
    # Get the training and validation sets
    X_train_subset = X_train[:i]
    y_train_subset = y_train[:i]

    # Fit the linear regression model
    lin_reg.fit(X_train_subset, y_train_subset)

    # Compute the training and validation errors
    train_error = mean_squared_error(y_train_subset, lin_reg.predict(X_train_subset))
    val_error = mean_squared_error(y_val, lin_reg.predict(X_val))

    # Append the errors to the lists
    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(10, 61, 10), train_errors, label='Training error')
plt.plot(range(10, 61, 10), val_errors, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve (Linear Regression)')
plt.legend()
plt.show()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the polynomial regression model
poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.transform(X_val)

poly_reg = LinearRegression()

# Initialize lists to store the training and validation errors
train_errors = []
val_errors = []

# Iterate over different training set sizes
for i in range(10, 61, 10):
    # Get the training and validation sets
    X_train_subset = X_train_poly[:i]
    y_train_subset = y_train[:i]

    # Fit the polynomial regression model
    poly_reg.fit(X_train_subset, y_train_subset)

    # Compute the training and validation errors
    train_error = mean_squared_error(y_train_subset, poly_reg.predict(X_train_subset))
    val_error = mean_squared_error(y_val, poly_reg.predict(X_val_poly))

    # Append the errors to the lists
    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(10, 61, 10), train_errors, label='Training error')
plt.plot(range(10, 61, 10), val_errors, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve (Polynomial Regression, Degree 10)')
plt.legend()
plt.show()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the polynomial regression model
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.transform(X_val)

poly_reg = LinearRegression()

# Initialize lists to store the training and validation errors
train_errors = []
val_errors = []

# Iterate over different training set sizes
for i in range(10, 61, 10):
    # Get the training and validation sets
    X_train_subset = X_train_poly[:i]
    y_train_subset = y_train[:i]

    # Fit the polynomial regression model
    poly_reg.fit(X_train_subset, y_train_subset)

    # Compute the training and validation errors
    train_error = mean_squared_error(y_train_subset, poly_reg.predict(X_train_subset))
    val_error = mean_squared_error(y_val, poly_reg.predict(X_val_poly))

    # Append the errors to the lists
    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(10, 61, 10), train_errors, label='Training error')
plt.plot(range(10, 61, 10), val_errors, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve (Polynomial Regression, Degree 2)')
plt.legend()
plt.show()