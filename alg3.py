import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

X_scaled = scale(X)

# Разные варианты параметра метрики p
p_values = np.linspace(1, 10, 200)  # 200 вариантов от 1 до 10
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Оптимальное p
best_p = None
best_score = float('-inf')

for p in p_values:
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    
    if mean_score > best_score:
        best_score = mean_score
        best_p = p

# Вывод результатов
print(f"Оптимальное значение p: {best_p}, качество (среднеквадратичная ошибка): {-best_score}")