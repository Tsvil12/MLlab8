import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel('Lab 7/1_knn_data.xlsx')

print(df.head())

plt.scatter(df['x1'], df['x2'], c = df['class'])
plt.show()

X = df[['x1', 'x2']]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))