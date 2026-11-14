"""Intensity K-means with cluster-to-class mapping learned on training pixels."""
from itertools import permutations

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


class BasicKMeans:
    def __init__(self, num_classes=4, seed=42, temperature=0.02):
        if num_classes != 4 or temperature <= 0:
            raise ValueError('Four classes and a positive temperature are required')
        self.num_classes = num_classes
        self.temperature = temperature
        self.estimator = MiniBatchKMeans(n_clusters=num_classes, random_state=seed,
                                        n_init=3, batch_size=4096)

    def fit(self, pixels, labels):
        pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(pixels) != len(labels) or len(np.unique(pixels)) < self.num_classes:
            raise ValueError('K-means requires paired labels and at least four distinct intensities')
        clusters = self.estimator.fit_predict(pixels)
        confusion = np.bincount(clusters * self.num_classes + labels,
                                minlength=self.num_classes**2).reshape(4, 4)
        self.mapping = np.array(max(permutations(range(4)),
                                   key=lambda p: sum(confusion[i, p[i]] for i in range(4))))
        self.inertia = float(self.estimator.inertia_ / len(pixels))
        return self

    def __call__(self, images):
        # Softmax of negative squared centroid distances provides a declared
        # probabilistic cross-entropy diagnostic; it is not the K-means objective.
        centers = torch.as_tensor(self.estimator.cluster_centers_[:, 0],
                                  device=images.device, dtype=images.dtype)
        logits = -(images - centers[None, :, None, None]).square() / self.temperature
        inverse = torch.as_tensor(np.argsort(self.mapping), device=images.device)
        return logits[:, inverse]
