import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.PatchDataset import (IGNORE_INDEX, PatchDataset, discover_samples,
                                  split_samples)
from experiments import confusion_matrix, iou_from_confusion
from models.BasicCNN import BasicCNN
from models.BasicKmeans import BasicKMeans
from models.BasicUnet import BasicUNet


def make_fixture(root, count=4):
    """Small synthetic four-class fixture, not a sonar quality benchmark."""
    (root / 'image').mkdir()
    for name in ('0', '1', '7', '10'):
        (root / 'mask' / name).mkdir(parents=True)
    labels = np.tile(np.arange(28) % 4, (24, 1))
    for index in range(count):
        name = f'scan_{index}.png'
        raw = (labels * 60 + 20 + index).astype(np.uint8)
        Image.fromarray(raw).save(root / 'image' / name)
        for class_id, folder in enumerate(('0', '1', '7', '10')):
            Image.fromarray(((labels == class_id) * 255).astype(np.uint8)).save(
                root / 'mask' / folder / name)


class PatchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_fixture(self.root)
        self.samples, self.names = discover_samples(self.root)

    def test_folder_mapping_split_and_border_patches(self):
        self.assertEqual(self.names, ['0', '1', '7', '10'])
        train, validation = split_samples(self.samples, seed=2)
        self.assertFalse({s.image for s in train} & {s.image for s in validation})
        ds = PatchDataset(validation, 10)
        self.assertEqual(len(ds), 9)
        x, y = ds[-1]
        self.assertEqual(x.shape, (1, 10, 10))
        self.assertEqual((y != IGNORE_INDEX).sum().item(), 4 * 8)
        self.assertEqual(set(y[y != IGNORE_INDEX].unique().tolist()), {0, 1, 2, 3})
        self.assertTrue(torch.all(y[4:] == IGNORE_INDEX))
        with Image.open(validation[0].image) as original:
            expected = np.array(original.crop((20, 20, 28, 24)), dtype=np.float32) / 255
        np.testing.assert_allclose(x[0, :4, :8].numpy(), expected)

    def test_overlap_unlabelled_and_dimension_validation(self):
        path = self.samples[0].masks[0]
        Image.fromarray(np.ones((24, 28), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            PatchDataset(self.samples, 8)[0]
        for path in self.samples[0].masks:
            Image.fromarray(np.zeros((24, 28), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'Unlabelled'):
            PatchDataset(self.samples, 8)[0]
        _, y = PatchDataset(self.samples, 8, background_class=2)[0]
        self.assertTrue(torch.all(y == 2))
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            PatchDataset(self.samples, 8)

    def test_rgb_and_rgba_masks_match_single_channel_masks(self):
        dataset = PatchDataset(self.samples, 10)
        expected = [dataset[i][1].clone() for i in range(len(dataset))]
        for mode in ('RGB', 'RGBA'):
            with self.subTest(mode=mode):
                for sample in self.samples:
                    for path in sample.masks:
                        with Image.open(path) as mask:
                            converted = mask.convert(mode)
                        converted.save(path)
                for index, target in enumerate(expected):
                    self.assertTrue(torch.equal(dataset[index][1], target))

    def test_missing_mask_fails(self):
        self.samples[0].masks[2].unlink()
        with self.assertRaisesRegex(ValueError, 'Missing masks'):
            discover_samples(self.root)

    def test_metric_aggregation_and_ignored_pixels(self):
        target = torch.tensor([[[0, 0, 1, 1, IGNORE_INDEX]]])
        predicted = torch.tensor([[[0, 1, 1, 1, 3]]])
        matrix = confusion_matrix(target, predicted)
        per_class, mean = iou_from_confusion(matrix)
        np.testing.assert_allclose(per_class[:2], [0.5, 2/3])
        self.assertTrue(np.isnan(per_class[2:]).all())
        self.assertAlmostEqual(mean, (0.5 + 2/3) / 2)
        self.assertEqual(matrix.sum(), 4)

    def test_models_support_grayscale_odd_patch_and_backward(self):
        torch.set_num_threads(2)
        for model in (BasicCNN(), BasicUNet()):
            x = torch.rand(2, 1, 11, 13)
            logits = model(x)
            self.assertEqual(logits.shape, (2, 4, 11, 13))
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 11, 13, dtype=torch.long))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_kmeans_class_mapping(self):
        values = np.repeat([0.1, 0.3, 0.6, 0.9], 20).astype(np.float32)
        labels = np.repeat([2, 0, 3, 1], 20)
        model = BasicKMeans(seed=1).fit(values, labels)
        predictions = model(torch.tensor(values.reshape(1, 1, 8, 10))).argmax(1).numpy().ravel()
        np.testing.assert_array_equal(predictions, labels)

    def test_notebook_executes_all_five_experiments(self):
        import nbformat
        from nbclient import NotebookClient

        # Reproduce the user's black/white three-channel PNG mask layout.
        for path in (self.root / 'mask').glob('*/*.png'):
            with Image.open(path) as mask:
                converted = mask.convert('RGB')
            converted.save(path)
        repository = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(repository / 'patch_training_run.ipynb', as_version=4)
        config = notebook.cells[1]
        config.source = config.source.replace(
            'DATA_ROOT = REPO_ROOT / "data"', f'DATA_ROOT = Path({str(self.root)!r})'
        ).replace('PATCH_SIZES = (128, 256)', 'PATCH_SIZES = (8, 12)').replace(
            'EPOCHS = 5', 'EPOCHS = 1'
        ).replace('MAX_KMEANS_PIXELS = 200_000', 'MAX_KMEANS_PIXELS = 2000').replace(
            'DEVICE = None', 'DEVICE = "cpu"'
        )
        config.source += '\nimport torch\ntorch.set_num_threads(2)\n'
        notebook.cells.append(nbformat.v4.new_code_cell(
            'assert len(results) == 5\n'
            'for result in results.values():\n'
            '    assert len(result["class_iou"]) == 4\n'
            '    assert np.isfinite(result["validation_loss"])\n'
            '    assert 0 <= result["mean_iou"] <= 1\n'
        ))
        notebook.cells[-1].source = 'import numpy as np\n' + notebook.cells[-1].source
        NotebookClient(notebook, timeout=120, kernel_name='python3',
                       resources={'metadata': {'path': str(repository)}}).execute()
        experiments = [cell for cell in notebook.cells if cell.cell_type == 'code'
                       and ' = run_experiment(' in cell.source]
        self.assertEqual(len(experiments), 5)
        for cell in experiments:
            text = ''.join(out.get('text', '') for out in cell.outputs)
            self.assertIn('loss=', text)
            self.assertIn('IoU=', text)
            plots = [out for out in cell.outputs if 'image/png' in out.get('data', {})]
            self.assertEqual(len(plots), 1)


if __name__ == '__main__':
    unittest.main()
