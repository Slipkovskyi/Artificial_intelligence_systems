from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Перевірка розміру датасету
print(dataset.shape)  # Очікувано (150, 5)

# Перегляд перших 20 рядків
print(dataset.head(20))

# Статистичне резюме даних
print(dataset.describe())

# Розподіл за класами
print(dataset.groupby('class').size())

# Візуалізація: діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# Візуалізація: гістограма
dataset.hist()
pyplot.show()

# Візуалізація: матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.show()

# Розділення на навчальні та тестові дані
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Стратифікована 10-кратна крос-валідація
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

# Оцінка моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінка кожної моделі
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results.mean())
    names.append(name)
    print(f'{name}: {cv_results.mean()}')

# Оцінка найкращої моделі
best_index = results.index(max(results))
best_name = names[best_index]
best_model = models[best_index][1]
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)
print(f"\nBest Model: {best_name}")
print("Accuracy Score:", accuracy_score(Y_validation, predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("Classification Report:\n", classification_report(Y_validation, predictions))