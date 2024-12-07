# Імпорт бібліотек
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt

# Завантаження даних з файлу
# Припустимо, що стовпці розділені пробілами (корегуйте за необхідністю)
input_file = "income_data.txt"  # Замість цього використайте фактичний шлях до файлу
names = ['feature1', 'feature2', 'feature3', 'feature4', 'target']  # Назви стовпців

# Завантаження даних
dataset = pd.read_csv(input_file, delim_whitespace=True, header=None, names=names)

# Перегляд даних
print(dataset.head())

# Очищення даних від зайвих символів
dataset = dataset.replace(',', '', regex=True)

# Перетворення на числові значення (перевіряємо дані перед цим)
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Перевіряємо наявність пропусків
print(dataset.isnull().sum())

# Видаляємо рядки, які містять NaN
dataset = dataset.dropna()

# Перевірка розмірів після очищення
print(f"Розмір даних після очищення: {dataset.shape}")

# Якщо є хоча б кілька рядків, можемо продовжити далі
if not dataset.empty:
    # Розділяємо на ознаки і мітки
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Розділення на тренувальні та тестові дані
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Ініціалізація моделей
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    # Оцінка моделей
    results = []
    names = []
    scoring = 'accuracy'

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

    # Візуалізація порівняння
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()
else:
    print("Набір даних порожній після очищення!")