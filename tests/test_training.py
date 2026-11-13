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
from train import pair_files, run_epoch


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

    def binary_fixture(self):
        images_dir, masks_dir = self.root / 'img', self.root / 'mask'
        images_dir.mkdir()
        for class_id in (1, 7, 10):
            (masks_dir / str(class_id)).mkdir(parents=True)
        for name, height in (('01.png', 1500), ('02.png', 1600)):
            pixels = np.repeat((np.arange(height) % 256).astype(np.uint8)[:, None], 12, axis=1)
            Image.fromarray(pixels).save(images_dir / name)
            for class_id, column in ((1, 1), (7, 2), (10, 3)):
                mask = np.zeros((height, 12), dtype=np.uint8)
                mask[:, column] = 255
                mask[:, 4] = 255  # All foreground masks overlap here.
                # Distinct annotation below the spatial boundary verifies Y alignment.
                mask[int(height * 0.8):, column] = 0
                mask[int(height * 0.8):, column + 5] = 1
                Image.fromarray(mask).save(masks_dir / str(class_id) / name)
        return images_dir, masks_dir

    def test_binary_layout_crops_and_priority(self):
        images_dir, masks_dir = self.binary_fixture()
        images, masks = pair_files(images_dir, masks_dir)
        self.assertEqual([p.name for p in images], ['01.png', '02.png'])
        for height in (64, 128, 256):
            for split in ('train', 'test'):
                ds = NikelDataset(images, masks, height, split=split)
                self.assertEqual(ds.class_values, (0, 1, 7, 10))
                for index in range(len(ds)):
                    x, y = ds[index]
                    self.assertEqual(tuple(x.shape), (1, height, 12))
                    expected = ([0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0]
                                if split == 'train' else
                                [0, 0, 0, 0, 3, 0, 1, 2, 3, 0, 0, 0])
                    self.assertEqual(y[0].tolist(), expected)
                    self.assertEqual(y[-1].tolist(), expected)
        ds = NikelDataset(images, masks, class_priority=[10, 7, 1])
        self.assertEqual(ds[0][1][0, 4].item(), 1)

    def test_binary_layout_validation(self):
        images_dir, masks_dir = self.binary_fixture()
        images, masks = pair_files(images_dir, masks_dir)
        with self.assertRaisesRegex(ValueError, 'class_priority'):
            NikelDataset(images, masks, class_priority=[1, 7])
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(masks_dir / '1' / '01.png')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            NikelDataset(images, masks)
        (masks_dir / '7' / '02.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            pair_files(images_dir, masks_dir)

    def test_dynamic_classes(self):
        images_dir, masks_dir = self.binary_fixture()
        (masks_dir / '10').rename(masks_dir / '42')
        images, masks = pair_files(images_dir, masks_dir)
        ds = NikelDataset(images, masks, split='test')
        self.assertEqual(ds.class_values, (0, 1, 7, 42))
        loss, miou, _ = run_epoch(BasicCNN(num_classes=len(ds.class_values)),
                                  DataLoader(ds), torch.device('cpu'),
                                  num_classes=len(ds.class_values))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= miou <= 1)


if __name__ == '__main__':
    unittest.main()
