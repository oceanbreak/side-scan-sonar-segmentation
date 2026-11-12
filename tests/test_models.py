import unittest

import numpy as np
import torch
from torch.nn import functional as F

from models import BasicCNN, BasicUnet, BasicKmeans


class NeuralModelsTest(unittest.TestCase):
    def test_shapes_and_training(self):
        torch.set_num_threads(2)
        for model_type in (BasicCNN, BasicUnet):
            for channels in (1, 3):
                with self.subTest(model=model_type.__name__, channels=channels):
                    model = model_type(in_channels=channels)
                    x = torch.randn(2, channels, 17, 23)
                    target = torch.randint(0, 4, (2, 17, 23))
                    logits = model(x)
                    self.assertEqual(logits.shape, (2, 4, 17, 23))
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    before = model.head.weight.detach().clone() if isinstance(model, BasicUnet) else model.network[-1].weight.detach().clone()
                    F.cross_entropy(logits, target).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    optimizer.step()
                    after = model.head.weight if isinstance(model, BasicUnet) else model.network[-1].weight
                    self.assertFalse(torch.equal(before, after))

    def test_unet_minimum_shape(self):
        self.assertEqual(BasicUnet()(torch.randn(1, 1, 4, 5)).shape, (1, 4, 4, 5))
        with self.assertRaises(ValueError):
            BasicUnet()(torch.randn(1, 1, 3, 8))


class KmeansTest(unittest.TestCase):
    def test_local_statistics(self):
        image = np.arange(25).reshape(5, 5)
        features = BasicKmeans(window_size=3).extract_features(image)
        expected = image[1:4, 1:4]
        np.testing.assert_allclose(features[2, 2], [expected.mean(), expected.std(), np.median(expected)])
        np.testing.assert_allclose(BasicKmeans().extract_features(np.ones((2, 3))), np.tile([1, 0, 1], (2, 3, 1)))

    def test_segmentation_and_repeatability(self):
        image = np.zeros((20, 30))
        image[:, 15:] = 100
        model = BasicKmeans(n_clusters=2, window_size=1, max_samples=100)
        labels = model.fit_predict(image)
        self.assertEqual(labels.shape, image.shape)
        self.assertEqual(len(np.unique(labels[:, :15])), 1)
        self.assertEqual(len(np.unique(labels[:, 15:])), 1)
        self.assertNotEqual(labels[0, 0], labels[0, -1])
        np.testing.assert_array_equal(labels, model.predict(image))
        np.testing.assert_array_equal(labels, BasicKmeans(n_clusters=2, window_size=1, max_samples=100).fit_predict(image))

    def test_constant_and_invalid_inputs(self):
        labels = BasicKmeans().fit_predict(np.ones((4, 4)))
        self.assertTrue(np.isfinite(labels).all())
        with self.assertRaises(RuntimeError):
            BasicKmeans().predict(np.ones((4, 4)))
        for image in (np.zeros((2, 2, 3)), np.array([[np.nan]]), np.empty((0, 0))):
            with self.assertRaises(ValueError):
                BasicKmeans().fit(image)
        with self.assertRaises(ValueError):
            BasicKmeans(window_size=2)
        with self.assertRaises(ValueError):
            BasicKmeans(n_clusters=5).fit(np.ones((2, 2)))


if __name__ == '__main__':
    unittest.main()
