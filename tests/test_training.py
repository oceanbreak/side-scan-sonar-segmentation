import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset.SegmentationDataset import SegmentationDataset, discover_pairs
from models import BasicCNN
from train import fit


class TrainingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.masks = self.root / 'masks'
        self.images.mkdir()
        self.masks.mkdir()
        for i in range(3):
            Image.fromarray(np.full((16, 12), 50 + i * 50, np.uint8)).save(self.images / f'{i}.png')
            mask = np.tile(np.array([1, 2, 3, 4], np.uint8), (16, 3))
            Image.fromarray(mask).save(self.masks / f'{i}.png')
        torch.set_num_threads(2)

    def test_patches_and_mapping(self):
        data = SegmentationDataset(discover_pairs(self.images, self.masks), patch_height=8)
        self.assertEqual(len(data), 6)
        x, y = data[5]
        self.assertEqual(x.shape, (1, 8, 12))
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(set(y.unique().tolist()), {0, 1, 2, 3})
        self.assertAlmostEqual(x[0, 0, 0].item(), 150 / 127.5 - 1, places=6)
        resized = SegmentationDataset(data.pairs, resize=(9, 7))
        self.assertEqual(resized[0][1].shape, (9, 7))
        self.assertEqual(set(resized[0][1].unique().tolist()), {0, 1, 2, 3})

    def test_invalid_pairs_and_labels(self):
        (self.masks / '2.png').unlink()
        with self.assertRaises(ValueError):
            discover_pairs(self.images, self.masks)
        bad = self.masks / '0.png'
        Image.fromarray(np.full((16, 12), 255, np.uint8)).save(bad)
        with self.assertRaises(ValueError):
            SegmentationDataset([(self.images / '0.png', bad)])[0]

    def test_fit_updates_weights_and_saves(self):
        loader = DataLoader(SegmentationDataset(discover_pairs(self.images, self.masks)), batch_size=2)
        model = BasicCNN()
        before = model.network[0].weight.detach().clone()
        output = self.root / 'fit'
        history = fit(model, loader, epochs=2, device='cpu', output_dir=output, plot=True, progress=False)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(history['val_loss'], [])
        self.assertFalse(torch.equal(before, model.network[0].weight))
        checkpoint = torch.load(output / 'last.pt', weights_only=True)
        self.assertEqual(checkpoint['epoch'], 2)
        self.assertEqual(checkpoint['history'], json.loads((output / 'history.json').read_text()))
        self.assertTrue((output / 'best.pt').is_file())
        self.assertGreater((output / 'loss.png').stat().st_size, 100)

    def test_cli_both_models_with_validation(self):
        for model in ('cnn', 'unet'):
            output = self.root / model
            result = subprocess.run([
                sys.executable, 'train.py', '--images', str(self.images), '--masks', str(self.masks),
                '--model', model, '--epochs', '1', '--resize', '9', '13', '--patch-height', '8',
                '--device', 'cpu', '--output-dir', str(output), '--plot',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual(len(history['val_loss']), 1)
            self.assertIn('4 training patches, 2 validation patches', result.stdout)
            self.assertIn('100%', result.stderr)
            self.assertTrue((output / 'loss.png').is_file())


class BinaryMasksTest(TrainingTest):
    def setUp(self):
        super().setUp()
        for path in self.masks.iterdir():
            path.unlink()
        for name in ('10', '2', '1'):
            folder = self.masks / name
            folder.mkdir()
            for i in range(3):
                mask = np.zeros((16, 12), np.uint8)
                # Overlap at column 4; class 10 takes priority over class 2.
                if name == '1':
                    mask[:, 1:3] = 255
                elif name == '2':
                    mask[:, 3:5] = 255
                else:
                    mask[:, 4:6] = 255
                Image.fromarray(mask).save(folder / f'{i}.png')

    # Binary fixtures differ from the parent tests' class-index fixtures.
    def test_patches_and_mapping(self):
        pairs = discover_pairs(self.images, self.masks)
        data = SegmentationDataset(pairs, patch_height=8)
        self.assertEqual(data.class_names, ['background', '1', '2', '10'])
        self.assertEqual(data.num_classes, 4)
        self.assertEqual(len(data), 6)
        x, target = data[5]
        self.assertEqual(x.shape, (1, 8, 12))
        np.testing.assert_array_equal(target[0].numpy(), [0, 1, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0])
        resized = SegmentationDataset(pairs, resize=(8, 24))
        self.assertEqual(set(resized[0][1].unique().tolist()), {0, 1, 2, 3})

    def test_invalid_pairs_and_labels(self):
        (self.masks / '2' / '1.png').unlink()
        with self.assertRaisesRegex(ValueError, 'for class 2'):
            discover_pairs(self.images, self.masks)

    def test_binary_cli_class_count(self):
        # Two foreground classes plus background must produce THREE channels.
        for path in (self.masks / '10').iterdir():
            path.unlink()
        (self.masks / '10').rmdir()
        output = self.root / 'binary-cli'
        result = subprocess.run([
            sys.executable, 'train.py', '--images', str(self.images), '--masks', str(self.masks),
            '--epochs', '1', '--resize', '8', '12', '--device', 'cpu',
            '--output-dir', str(output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = json.loads((output / 'config.json').read_text())
        self.assertEqual(config['class_names'], ['background', '1', '2'])
        self.assertEqual(config['num_classes'], 3)
        checkpoint = torch.load(output / 'last.pt', weights_only=True)
        self.assertEqual(checkpoint['model_state_dict']['network.6.weight'].shape[0], 3)

    def test_mask_size_mismatch(self):
        Image.fromarray(np.zeros((3, 4), np.uint8)).save(self.masks / '1' / '0.png')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            SegmentationDataset(discover_pairs(self.images, self.masks))


if __name__ == '__main__':
    unittest.main()
