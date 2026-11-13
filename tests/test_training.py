import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.NickelDataset import NikelDataset
from metrics import compute_iou, update_confusion
from models.BasicCNN import BasicCNN
from torch.utils.data import DataLoader
from train import run_epoch


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images, self.masks = [], []
        for i, height in enumerate((1500, 1600)):
            image, mask = self.root / f'image{i}.png', self.root / f'mask{i}.png'
            rows = (np.arange(height) % 256).astype(np.uint8)
            Image.fromarray(np.repeat(rows[:, None], 12, axis=1)).save(image)
            labels = np.array([0, 1, 7, 10], dtype=np.uint8)
            Image.fromarray(np.tile(labels, (height, 3))).save(mask)
            self.images.append(image)
            self.masks.append(mask)

    def test_spatial_split_and_patches(self):
        for height in (64, 128, 256):
            with self.subTest(height=height):
                train = NikelDataset(self.images, self.masks, height, split='train')
                test = NikelDataset(self.images, self.masks, height, split='test')
                for ds in (train, test):
                    for index in range(len(ds)):
                        source, box = ds.patch_location(index)
                        x, y = ds[index]
                        self.assertEqual(tuple(x.shape), (1, height, 12))
                        self.assertEqual(tuple(y.shape), (height, 12))
                        self.assertEqual(y[0].tolist(), [0, 1, 2, 3] * 3)
                        expected = (box[1] % 256) / 127.5 - 1
                        self.assertAlmostEqual(x[0, 0, 0].item(), expected, places=5)
                        boundary = train.regions[source][1]
                        self.assertTrue(box[3] <= boundary if ds is train else box[1] >= boundary)
                self.assertEqual(train.patch_location(-1), train.patch_location(len(train)-1))
                with self.assertRaises(IndexError):
                    train[len(train)]

    def test_defaults_and_validation(self):
        self.assertEqual(NikelDataset(self.images, self.masks).patch_height, 128)
        with self.assertRaises(ValueError):
            NikelDataset(self.images, self.masks, 0)
        with self.assertRaises(ValueError):
            NikelDataset(self.images, [], 128)
        with self.assertRaises(ValueError):
            NikelDataset(self.images, self.masks, 512, split='test')
        bad = self.root / 'bad.png'
        Image.fromarray(np.full((1500, 12), 255, dtype=np.uint8)).save(bad)
        with self.assertRaisesRegex(ValueError, 'Unknown mask'):
            NikelDataset(self.images[:1], [bad])[0]

    def test_iou_known_values_and_absent_classes(self):
        confusion = torch.zeros((4, 4), dtype=torch.int64)
        update_confusion(confusion, torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
        mean, per_class = compute_iou(confusion)
        self.assertAlmostEqual(per_class[0], 1/2)
        self.assertAlmostEqual(per_class[1], 2/3)
        self.assertAlmostEqual(mean, (1/2 + 2/3)/2)
        self.assertTrue(np.isnan(per_class[2]))
        mean, _ = compute_iou(torch.eye(4, dtype=torch.int64))
        self.assertEqual(mean, 1)

    def test_training_and_evaluation(self):
        ds = NikelDataset(self.images[:1], self.masks[:1], 256, split='test')
        model = BasicCNN()
        optimizer = torch.optim.Adam(model.parameters())
        before = [p.detach().clone() for p in model.parameters()]
        loss, miou, _ = run_epoch(model, DataLoader(ds), torch.device('cpu'), optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= miou <= 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        before = [p.detach().clone() for p in model.parameters()]
        run_epoch(model, DataLoader(ds), torch.device('cpu'))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))


if __name__ == '__main__':
    unittest.main()
