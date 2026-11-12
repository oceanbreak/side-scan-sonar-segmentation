"""NumPy K-means segmentation using local mean, std and median features."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class BasicKmeans:
    """Cluster a finite 2D grayscale image; labels are not semantic class IDs.

    Fit on one representative image, then predict others on the same intensity
    scale. Training uses at most max_samples pixels; prediction covers all pixels.
    """

    def __init__(self, n_clusters=4, window_size=5, max_iter=50,
                 max_samples=10000, random_state=0, tol=1e-4):
        for name, value in (("n_clusters", n_clusters), ("window_size", window_size),
                            ("max_iter", max_iter), ("max_samples", max_samples)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if window_size % 2 != 1:
            raise ValueError("window_size must be odd")
        if max_samples < n_clusters or not np.isfinite(tol) or tol < 0:
            raise ValueError("Require max_samples >= n_clusters and finite tol >= 0")
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.max_iter = max_iter
        self.max_samples = max_samples
        self.random_state = random_state
        self.tol = tol

    def extract_features(self, image):
        """Return (H, W, 3) local mean/std/median, with replicated borders."""
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
            raise ValueError("Expected a nonempty finite 2D grayscale image")
        radius = self.window_size // 2
        padded = np.pad(image, radius, mode="edge")
        windows = sliding_window_view(padded, (self.window_size, self.window_size))
        features = np.empty(image.shape + (3,), dtype=np.float64)
        # Bound temporary reductions instead of materializing all image windows.
        for row in range(image.shape[0]):
            local = windows[row]
            features[row, :, 0] = local.mean(axis=(-2, -1))
            features[row, :, 1] = local.std(axis=(-2, -1))
            features[row, :, 2] = np.median(local, axis=(-2, -1))
        return features

    @staticmethod
    def _assign(features, centers):
        labels = np.empty(len(features), dtype=np.int64)
        for start in range(0, len(features), 8192):
            chunk = features[start:start + 8192]
            distances = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels[start:start + len(chunk)] = distances.argmin(axis=1)
        return labels

    def fit(self, image):
        features = self.extract_features(image).reshape(-1, 3)
        if len(features) < self.n_clusters:
            raise ValueError("Image must contain at least n_clusters pixels")
        rng = np.random.default_rng(self.random_state)
        sample = features[rng.choice(len(features),
                                     min(len(features), self.max_samples), replace=False)]
        mean = sample.mean(axis=0)
        scale = sample.std(axis=0)
        scale[scale == 0] = 1
        sample = (sample - mean) / scale
        centers = sample[rng.choice(len(sample), self.n_clusters, replace=False)].copy()
        for _ in range(self.max_iter):
            labels = self._assign(sample, centers)
            updated = centers.copy()
            for cluster in range(self.n_clusters):
                members = sample[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = sample[rng.integers(len(sample))]
            shift = np.linalg.norm(updated - centers)
            centers = updated
            if shift <= self.tol:
                break
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.cluster_centers_ = centers
        return self

    def predict(self, image):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Call fit before predict")
        features = self.extract_features(image)
        normalized = (features.reshape(-1, 3) - self.feature_mean_) / self.feature_scale_
        return self._assign(normalized, self.cluster_centers_).reshape(features.shape[:2])

    def fit_predict(self, image):
        return self.fit(image).predict(image)
