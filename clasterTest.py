import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import warnings

import hw

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# df = sns.load_dataset("iris")
# #pd.DataFrame(sklearn.datasets.load_iris) #
# print(df.head())
# print(df["species"].unique())
# print(df.describe().T)

# Генерация данных (например, полукруги)
#X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X=np.array(hw.for_cluster)

# Настройка параметров DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=1)

# Применение DBSCAN к данным
clusters = dbscan.fit_predict(X)

# Визуализация кластеров
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()