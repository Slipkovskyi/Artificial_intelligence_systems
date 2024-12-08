# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантаження набору даних про житло в Каліфорнії
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Перемішування даних для підвищення надійності моделі
X, y = shuffle(X, y, random_state=7)

# Розподіл даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Ініціалізація регресора AdaBoost з базовим деревом ухвалення рішень
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),  # Глибина дерева
    n_estimators=400,  # Кількість дерев у ансамблі
    random_state=7
)

# Навчання моделі на тренувальних даних
regressor.fit(X_train, y_train)

# Прогнозування значень для тестових даних
y_pred = regressor.predict(X_test)

# Обчислення метрик якості моделі
mse = mean_squared_error(y_test, y_pred)  # Середньоквадратична помилка
evs = explained_variance_score(y_test, y_pred)  # Оцінка поясненої дисперсії

# Вивід метрик
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Обчислення важливості ознак
feature_importances = regressor.feature_importances_
feature_names = housing.feature_names

# Перевірка, чи збігаються довжини масивів
assert len(feature_importances) == len(feature_names), "Довжини масивів неспівпадають!"

# Нормалізація важливості ознак
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування індексів ознак за важливістю
index_sorted = np.flipud(np.argsort(feature_importances))

# Побудова гістограми важливості ознак
pos = np.arange(index_sorted.shape[0]) + 0.5
plt.figure(figsize=(12, 8))  # Установка ширини та висоти графіка
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, np.array(feature_names)[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Оцінка важливості ознак з використанням регресора AdaBoost')
plt.show()
