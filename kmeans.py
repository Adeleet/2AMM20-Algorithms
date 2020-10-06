from math import sqrt
from random import choices
from time import time

import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib.animation import PillowWriter
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.8,
    random_state=100,
)


class kmeans:
    def __init__(self, k) -> None:
        self.centroids = []
        self.k = k
        self.camera = Camera(plt.figure())
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def fit(self, X, eps=1e-6):
        self.C = [None] * len(X)
        self.centroids = dict(
            zip(list(range(self.k)), kmeans._pick_initial_clusters_(X, self.k),)
        )
        i = 0
        start_time = time()
        while True:
            self._assign_to_nearest_cluster_(X)
            cluster_movement = self._recompute_cluster_centers_()
            self._plot_()
            i += 1
            if cluster_movement < eps:
                break
        anim = self.camera.animate(blit=True)
        anim.save("scatter.gif", writer=PillowWriter(fps=5))
        print(f"[SUMMARY] Total within-cluster distance: {self._compute_totals_()}")
        print(f"[SUMMARY] Performed {i} iterations in {time()-start_time:.3f} seconds")

    def _pick_initial_clusters_(X, k):
        return choices(X, k=k)

    def _distance_(x1, x2):
        assert len(x1) == len(x2)
        return sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1))]))

    def _assign_to_nearest_cluster_(self, X):
        for i, obs in enumerate(X):
            distances = [
                (idx, kmeans._distance_(obs, c)) for idx, c in self.centroids.items()
            ]
            closest_centroid_idx = min(distances, key=lambda x: x[1])[0]
            self.C[i] = (closest_centroid_idx, obs)

    def _recompute_cluster_centers_(self):
        cluster_movements = []
        for idx, centroid in self.centroids.items():
            cluster_points = [x[1] for x in self.C if x[0] == idx]
            new_centroid = kmeans._mean_vector_(cluster_points)
            cluster_movements.append(kmeans._distance_(centroid, new_centroid))
            self.centroids[idx] = new_centroid
        return sum(cluster_movements)

    def _mean_vector_(v):
        return [sum([x[j] for x in v]) / len(v) for j in range(len(v[0]))]

    def _plot_(self):
        for k in range(self.k):
            x = [x[1][0] for x in self.C if x[0] == k]
            y = [x[1][1] for x in self.C if x[0] == k]
            plt.scatter(x, y, c=self.colors[k])
        x_centroids = [x[0] for x in self.centroids.values()]
        y_centroids = [x[1] for x in self.centroids.values()]
        plt.scatter(
            x_centroids, y_centroids, marker="x", color="black",
        )
        self.camera.snap()

    def _compute_totals_(self):
        W_C = 0
        for idx, centroid in self.centroids.items():
            cluster_points = [x[1] for x in self.C if x[0] == idx]
            W_C += (
                sum(
                    [
                        sum([kmeans._distance_(xA, xB) for xA in cluster_points])
                        for xB in cluster_points
                    ]
                )
                / 2
            )
        return W_C


km = kmeans(2)
km.fit(X)
