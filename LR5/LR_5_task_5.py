import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Вказуємо шлях до файлу з даними
input_file = 'traffic_data.txt'
data = []

# Читання даних з файлу та збереження їх у списку
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')  # Розділення кожного рядка на елементи
        data.append(items)

# Перетворення даних у масив numpy
data = np.array(data)

label_encoder = []  # Список для зберігання лейбл-кодерів
X_encoded = np.empty(data.shape)  # Масив для збереження закодованих даних

# Перебір кожного стовпця для кодування
for i, item in enumerate(data[0]):
    if item.isdigit():  # Якщо значення числове
        X_encoded[:, i] = data[:, i]  # Просто переносимо дані без змін
    else:  # Якщо значення не числове (категоріальне)
        le = preprocessing.LabelEncoder()  # Створюємо об'єкт LabelEncoder
        X_encoded[:, i] = le.fit_transform(data[:, i])  # Перетворюємо категорії в числа
        label_encoder.append(le)  # Додаємо лейбл-кодер до списку

# Виділяємо вхідні дані (X) та мітки (y), останній стовпець є міткою
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розподіляємо дані на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Налаштування параметрів моделі Extra Trees Regressor
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)

# Навчання моделі на тренувальних даних
regressor.fit(X_train, y_train)

# Прогнозування для тестових даних
y_pred = regressor.predict(X_test)

# Виведення середньої абсолютної помилки для прогнозу
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Тестова точка для прогнозування
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [1] * len(test_datapoint)  # Створюємо порожній список для закодованих значень
count = 0

# Перетворення тестової точки на числові значення
for i, item in enumerate(test_datapoint):
    if item.isdigit():  # Якщо значення числове
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:  # Якщо значення категоріальне
        test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])[0])  # Використовуємо лейбл-кодер
    count += 1

# Перетворення тестових даних в numpy масив
test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозування для тестової точки та виведення результату
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))
