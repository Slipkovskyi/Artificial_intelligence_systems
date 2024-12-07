import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Загрузка входных данных
data = np.loadtxt('data_clustering.txt', delimiter=',')

# Оценка ширины окна
bandwidth = estimate_bandwidth(data, quantile=0.2)

# Кластеризация методом сдвига среднего
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(data)

# Получение центров кластеров и меток
cluster_centers = mean_shift.cluster_centers_
labels = mean_shift.labels_
n_clusters = len(cluster_centers)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c='red', marker='x', s=200, linewidths=3)
plt.title(f'Кластеризация методом сдвига среднего (кол-во кластеров: {n_clusters})')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()

print(f"Количество кластеров: {n_clusters}")
print("Центры кластеров:")
print(cluster_centers)