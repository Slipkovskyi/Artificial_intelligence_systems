import sys  # Робота з аргументами командного рядка
import numpy as np  # Операції з масивами
import matplotlib.pyplot as plt  # Візуалізація графіків
from sklearn.ensemble import ExtraTreesClassifier  # Класифікатор ExtraTrees
from sklearn.model_selection import train_test_split  # Розбиття на навчальні та тестові дані
from sklearn.metrics import classification_report  # Оцінка якості моделі


# Функція для візуалізації результатів класифікації
def visualize_classifier(classifier, X, y, title=''):
    plt.figure()
    # Визначення меж області
    X_min, X_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Створення сітки координат
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Прогнозування класів на основі координат сітки
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Відображення областей класів
    plt.contourf(xx, yy, Z, alpha=0.8)
    # Відображення точок даних
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.title(title)
    plt.show()


# Завантаження вхідних даних з файлу
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]  # Розбиття на ознаки та мітки класів

# Візуалізація вхідних даних
class_0 = np.array(X[y == 0])  # Дані класу 0
class_1 = np.array(X[y == 1])  # Дані класу 1
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Вхідні дані')

# Розбиття на навчальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Налаштування параметрів класифікатора
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

# Обробка аргументу командного рядка (опціональне балансування класів)
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params['class_weight'] = 'balanced'
    else:
        raise TypeError("Invalid input argument; should be 'balance'")

# Навчання класифікатора
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

# Візуалізація результатів на навчальному наборі
visualize_classifier(classifier, X_train, y_train, 'Навчальний набір даних')

# Прогнозування та візуалізація на тестовому наборі
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Тестовий набір даних')

# Виведення звіту про якість моделі
class_names = ['Class-0', 'Class-1']
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names, zero_division=0))
print("#" * 40 + "\n")
print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))
print("#" * 40 + "\n")

# Відображення всіх графіків
plt.show()
