"""Four-class, grayscale patches from per-class binary mask folders."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 4
IGNORE_INDEX = 255
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


@dataclass(frozen=True)
class Sample:
    image: Path
    masks: tuple


def _indexed_images(directory):
    if not directory.is_dir():
        raise ValueError(f'Missing image directory: {directory}')
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in files:
                raise ValueError(f'Duplicate image stem {path.stem!r} in {directory}')
            files[path.stem] = path
    return files


def discover_samples(root, image_folder='image'):
    """Match stems; sorted numeric folder names map to contiguous class IDs 0..3."""
    root = Path(root)
    mask_root = root / 'mask'
    if not mask_root.is_dir():
        raise ValueError(f'Missing mask directory: {mask_root}')
    folders = sorted((p for p in mask_root.iterdir() if p.is_dir()),
                     key=lambda p: int(p.name) if p.name.isdigit() else -1)
    if len(folders) != NUM_CLASSES or any(not p.name.isdigit() for p in folders):
        raise ValueError('mask/ must contain exactly four numeric class directories')
    if len({int(p.name) for p in folders}) != NUM_CLASSES:
        raise ValueError('Class directory numbers must be distinct')
    images = _indexed_images(root / image_folder)
    if not images:
        raise ValueError(f'No images found in {root / image_folder}')
    mask_files = [_indexed_images(folder) for folder in folders]
    samples = []
    for stem, image in images.items():
        missing = [folders[i].name for i, files in enumerate(mask_files) if stem not in files]
        if missing:
            raise ValueError(f'Missing masks for {image.name} in classes {missing}')
        samples.append(Sample(image, tuple(files[stem] for files in mask_files)))
    return samples, [folder.name for folder in folders]


def split_samples(samples, validation_fraction=0.2, seed=42):
    """Split source images before tiling, preventing overlapping train/validation data."""
    if len(samples) < 2:
        raise ValueError('At least two source images are required for a held-out split')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    order = np.random.default_rng(seed).permutation(len(samples))
    count = min(len(samples) - 1, max(1, round(len(samples) * validation_fraction)))
    return ([samples[i] for i in order[count:]], [samples[i] for i in order[:count]])


class PatchDataset(Dataset):
    """Nonoverlapping square tiles; padded/optionally unlabelled pixels are ignored.

    Masks must be single-channel binary images (zero/nonzero). Overlapping class
    masks are rejected. background_class=None rejects any unlabelled pixels;
    otherwise unlabelled pixels belong to the explicitly selected class ID.
    """
    def __init__(self, samples, patch_size, background_class=None):
        if not isinstance(patch_size, int) or patch_size < 4:
            raise ValueError('patch_size must be an integer >= 4')
        if background_class is not None and background_class not in range(NUM_CLASSES):
            raise ValueError('background_class must be None or a class ID from 0 to 3')
        self.samples = list(samples)
        self.patch_size = patch_size
        self.background_class = background_class
        self.tiles = []
        for index, sample in enumerate(self.samples):
            with Image.open(sample.image) as image:
                width, height = image.size
            for path in sample.masks:
                with Image.open(path) as mask:
                    if mask.size != (width, height):
                        raise ValueError(f'Mask dimensions differ from image: {path}')
            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    self.tiles.append((index, left, top, min(patch_size, width-left),
                                       min(patch_size, height-top)))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        source, left, top, width, height = self.tiles[index]
        sample = self.samples[source]
        box = (left, top, left+width, top+height)
        with Image.open(sample.image) as image:
            # The documented SSS inputs are 8-bit single-channel images.
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Expected 8-bit image, got {image.mode}: {sample.image}')
            raw = np.array(image.crop(box).convert('L'), dtype=np.float32) / 255.0
        masks = []
        for path in sample.masks:
            with Image.open(path) as mask:
                values = np.array(mask.crop(box))
            if values.ndim != 2:
                raise ValueError(f'Expected single-channel binary mask: {path}')
            masks.append(values != 0)
        masks = np.stack(masks)
        coverage = masks.sum(axis=0)
        if (coverage > 1).any():
            raise ValueError(f'Overlapping class masks in {sample.image.name}, tile {box}')
        if self.background_class is None and (coverage == 0).any():
            raise ValueError('Unlabelled pixels: set background_class explicitly if appropriate')
        labels = masks.argmax(axis=0).astype(np.int64)
        if self.background_class is not None:
            labels[coverage == 0] = self.background_class
        image_tile = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)
        target = np.full((self.patch_size, self.patch_size), IGNORE_INDEX, dtype=np.int64)
        image_tile[0, :height, :width] = raw
        target[:height, :width] = labels
        return torch.from_numpy(image_tile), torch.from_numpy(target)
