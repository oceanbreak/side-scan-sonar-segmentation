"""Grayscale images with class-index masks or per-class binary masks."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _collect(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Not a directory: ' + str(directory))
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.suffix.lower() in EXTENSIONS and path.is_file():
            key = path.relative_to(directory).with_suffix('').as_posix()
            if key in files:
                raise ValueError('Duplicate image stem: ' + key)
            files[key] = path
    return files


def _check_stems(images, masks, context=''):
    if not images or images.keys() != masks.keys():
        raise ValueError('Need nonempty matching image/mask stems ' + context
                         + '; missing masks: ' + str(sorted(images.keys() - masks.keys()))
                         + '; missing images: ' + str(sorted(masks.keys() - images.keys())))


def discover_pairs(images_dir, masks_dir):
    """Auto-detect MASKS/<class>/<stem> or flat class-index masks.

    Class-folder pairs contain an ordered {class_name: mask_path} dictionary.
    Numeric class folder names sort numerically; other names alphabetically.
    """
    images = _collect(images_dir)
    root = Path(masks_dir)
    if not root.is_dir():
        raise ValueError('Not a directory: ' + str(root))
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    direct_masks = any(path.is_file() and path.suffix.lower() in EXTENSIONS
                       for path in root.iterdir())
    if folders and not direct_masks:
        if all(path.name.isdigit() for path in folders):
            folders.sort(key=lambda path: (int(path.name), path.name))
        by_class = {folder.name: _collect(folder) for folder in folders}
        for name, masks in by_class.items():
            _check_stems(images, masks, 'for class ' + name)
        return [(images[key], {name: masks[key] for name, masks in by_class.items()})
                for key in sorted(images)]
    masks = _collect(masks_dir)
    _check_stems(images, masks)
    return [(images[key], masks[key]) for key in sorted(images)]


class SegmentationDataset(Dataset):
    """Load grayscale image patches with semantic masks.

    Binary dictionaries map classes to paths. Background is 0, dictionary order
    maps classes to 1..N, and later classes win overlaps. Nonzero mask pixels are
    foreground. All pairs must use the same dictionary order.
    For class-index masks, class_values maps raw values to consecutive IDs.
    Bottom remainders smaller than patch_height are discarded.
    resize=(height, width) uses bilinear images and nearest-neighbor masks.
    """
    def __init__(self, pairs, class_values=(1, 2, 3, 4), patch_height=None,
                 resize=None):
        self.pairs = list(pairs)
        self.class_values = tuple(class_values)
        if not self.pairs or not self.class_values or len(set(self.class_values)) != len(self.class_values):
            raise ValueError('Need image pairs and unique class_values')
        if patch_height is not None and (not isinstance(patch_height, int) or patch_height < 1):
            raise ValueError('patch_height must be a positive integer')
        if resize is not None and (len(resize) != 2 or any(v < 1 for v in resize)):
            raise ValueError('resize must contain positive height and width')
        self.binary_masks = isinstance(self.pairs[0][1], dict)
        self.class_names = (['background'] + list(self.pairs[0][1]) if self.binary_masks
                            else [str(value) for value in self.class_values])
        if self.binary_masks and len(self.class_names) == 1:
            raise ValueError('Need at least one binary-mask class')
        self.num_classes = len(self.class_names)
        self.resize = resize
        self.patches = []
        for index, (image_path, mask_path) in enumerate(self.pairs):
            if isinstance(mask_path, dict) != self.binary_masks:
                raise ValueError('Cannot mix binary and class-index masks')
            if self.binary_masks and list(mask_path) != self.class_names[1:]:
                raise ValueError('All pairs must have identical ordered mask classes')
            with Image.open(image_path) as image:
                width, height = image.size
            paths = mask_path.values() if self.binary_masks else [mask_path]
            for path in paths:
                with Image.open(path) as mask:
                    if (width, height) != mask.size:
                        raise ValueError('Image/mask size mismatch: ' + str(path))
            step = patch_height or height
            for top in range(0, height - step + 1, step):
                self.patches.append((index, (0, top, width, top + step)))
        if not self.patches:
            raise ValueError('No complete patches; reduce patch_height')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        pair_index, box = self.patches[index]
        image_path, mask_path = self.pairs[pair_index]
        with Image.open(image_path) as source:
            image = source.convert('L').crop(box)
        size = (self.resize[1], self.resize[0]) if self.resize is not None else None
        if size is not None:
            image = image.resize(size, Image.Resampling.BILINEAR)

        def read_mask(path, binary=False):
            with Image.open(path) as source:
                mask = (source.convert('L') if binary else source).crop(box)
            if size is not None:
                mask = mask.resize(size, Image.Resampling.NEAREST)
            raw = np.asarray(mask)
            if raw.ndim != 2:
                raise ValueError('Class-index masks must be single-channel images')
            return raw

        if self.binary_masks:
            target = np.zeros((image.height, image.width), dtype=np.int64)
            for class_id, path in enumerate(mask_path.values(), start=1):
                target[read_mask(path, binary=True) > 0] = class_id
        else:
            raw = read_mask(mask_path)
            target = np.full(raw.shape, -1, dtype=np.int64)
            for class_id, value in enumerate(self.class_values):
                target[raw == value] = class_id
            if (target < 0).any():
                raise ValueError('Unknown mask values: ' + str(np.unique(raw[target < 0]).tolist()))
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(pixels[None]), torch.from_numpy(target)
