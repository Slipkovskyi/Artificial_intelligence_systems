import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv'
data = pd.read_csv(url)

# Предварительная обработка данных
# Удаляем строки с пропущенными значениями
data = data.dropna()

# Преобразуем текстовые данные в числовые
data['destination'] = data['destination'].astype('category').cat.codes
data['origin'] = data['origin'].astype('category').cat.codes
data['train_type'] = data['train_type'].astype('category').cat.codes

# Создание категорий цен
bins = [0, 25, 50, 75, 100, 125, 150, 200, 250]
labels = ['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-200', '200-250']
data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels)

# Разделение данных на признаки и целевую переменную
X = data[['destination', 'origin', 'train_type']]
y = data['price_category']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Применение байєсівського классификатора
nb = GaussianNB()
nb.fit(X_train, y_train)

# Прогнозирование
y_pred = nb.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
