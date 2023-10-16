from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

iris = load_iris()
X = iris['data']
y = iris['target']

# Створення об'єкту КМеаns, навчання і передбачення вихідних міток
kmeans = KMeans(init='k-means++', n_clusters=y.max() + 1, n_init=10, max_iter=300, tol=0.0001,
                verbose=0, random_state=None, copy_x=True)
y_kmeans = kmeans.fit_predict(X)

# Графічне відображення вхідних точок і центрів кластеризації
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=50, c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


# Функція для знаходження кластерів
def find_clusters(X_, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X_.shape[0])[:n_clusters]
    centers_ = X_[i]

    while True:
        labels_ = pairwise_distances_argmin(X_, centers_)
        new_centers = np.array([X_[labels_ == i].mean(0) for i in range(n_clusters)])
        if np.all(centers_ == new_centers):
            break
        centers_ = new_centers

    return centers_, labels_


centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

labels = KMeans(3, random_state=0, n_init=10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
