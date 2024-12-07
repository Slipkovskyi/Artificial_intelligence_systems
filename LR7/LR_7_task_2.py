import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode


# Загрузка данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризация с помощью K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Визуализация результатов кластеризации
plt.figure(figsize=(12, 5))

# Первые два признака
plt.subplot(121)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Кластеризация Iris (1-2 признаки)')
plt.xlabel('Стандартизированная длина чашолистка')
plt.ylabel('Стандартизированная ширина чашолистка')

# Третий и четвертый признаки
plt.subplot(122)
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=kmeans.labels_, cmap='viridis')
plt.title('Кластеризация Iris (3-4 признаки)')
plt.xlabel('Стандартизированная длина лепестка')
plt.ylabel('Стандартизированная ширина лепестка')

plt.tight_layout()
plt.show()

# Сравнение меток кластеризации с истинными метками
from sklearn.metrics import accuracy_score
from scipy.stats import mode

def get_cluster_labels(kmeans_labels, true_labels):
    cluster_labels = np.zeros_like(kmeans_labels)
    for i in range(3):
        mask = (kmeans_labels == i)
        cluster_labels[mask] = mode(true_labels[mask], keepdims=False)[0]
    return cluster_labels

cluster_labels = get_cluster_labels(kmeans.labels_, y)
print("Точность кластеризации:", accuracy_score(y, cluster_labels))