import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("Коефіцієнти регресії: ", regr.coef_)
print("Вільний член: ", regr.intercept_)
print("Коефіцієнт детермінації R2: ", r2_score(y_test, y_pred))
print("Середня абсолютна помилка (MAE): ", mean_absolute_error(y_test, y_pred))
print("Середньоквадратична помилка (MSE): ", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label='Прогноз')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ідеальна лінія')
plt.xlabel('Виміряно')
plt.ylabel('Передбачено')
plt.legend()
plt.title('Лінійна регресія: Виміряні vs. Передбачені значення')
plt.show()
