import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import numpy as np

data = pd.read_csv('Lab 7/2_wine.data.txt', header=None)

X = data.iloc[:, 1:].values  # Признаки (все столбцы, кроме первого)
y = data.iloc[:, 0].values    # Классы (первый столбец)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
k_values = range(1, 51)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    accuracies.append(np.mean(scores))

# Оптимальное k и его качество
optimal_k = k_values[np.argmax(accuracies)]
optimal_accuracy = max(accuracies)

print(f"Оптимальное k: {optimal_k}, качество: {optimal_accuracy}")

# Масштабирование признаков
X_scaled = scale(X)

scaled_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    scaled_accuracies.append(np.mean(scores))

# Оптимальное k и его качество после масштабирования
optimal_k_scaled = k_values[np.argmax(scaled_accuracies)]
optimal_accuracy_scaled = max(scaled_accuracies)

print(f"Оптимальное k после масштабирования: {optimal_k_scaled}, качество: {optimal_accuracy_scaled}")