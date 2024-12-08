import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

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
