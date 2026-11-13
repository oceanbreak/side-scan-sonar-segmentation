"""Full-width sonar patches, optionally restricted to a spatial train/test split."""
import operator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NikelDataset(Dataset):
    """Cut along Y only. Drop incomplete patches at each region's bottom edge.

    ``split=None`` uses whole images. Otherwise the top ``train_fraction`` of
    every image is train and the remaining bottom region is test. Splitting is
    done before patching, so no patch crosses the train/test boundary.
    Masks contain class IDs 0, 1, 7, 10, mapped to contiguous IDs 0..3.
    """

    class_values = (0, 1, 7, 10)

    def __init__(self, images, segmentation_masks, patch_height=128,
                 split=None, train_fraction=0.8):
        patch_height = operator.index(patch_height)
        if patch_height <= 0:
            raise ValueError('patch_height must be positive')
        if split not in (None, 'train', 'test'):
            raise ValueError('split must be None, train, or test')
        if not 0 < train_fraction < 1:
            raise ValueError('train_fraction must be between 0 and 1')
        self.images = list(images)
        self.segmentation_masks = list(segmentation_masks)
        if len(self.images) != len(self.segmentation_masks):
            raise ValueError('Each image needs one matching class-ID mask')
        self.patch_height = patch_height
        self.split = split
        self.train_fraction = train_fraction
        self.regions = []
        self.index_intervals = [0]
        for image_file, mask_file in zip(self.images, self.segmentation_masks):
            with Image.open(image_file) as image, Image.open(mask_file) as mask:
                width, height = image.size
                if mask.size != image.size:
                    raise ValueError(f'Image/mask size mismatch: {image_file}')
            boundary = int(height * train_fraction)
            start = boundary if split == 'test' else 0
            stop = boundary if split == 'train' else height
            count = (stop - start) // patch_height
            if split is not None and count == 0:
                raise ValueError(
                    f'{image_file}: {split} region is too short for '
                    f'patch_height={patch_height}')
            self.regions.append((start, stop, width))
            self.index_intervals.append(self.index_intervals[-1] + count)
        self.index_intervals = np.asarray(self.index_intervals)
        self.total_patches_amount = int(self.index_intervals[-1])

    def patch_location(self, index):
        """Return (source image index, PIL crop box) for a dataset index."""
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('patch index out of range')
        image_index = int(np.searchsorted(self.index_intervals, index, side='right') - 1)
        patch_index = index - self.index_intervals[image_index]
        start, _, width = self.regions[image_index]
        top = int(start + patch_index * self.patch_height)
        return image_index, (0, top, width, top + self.patch_height)

    def _crop(self, paths, index):
        image_index, box = self.patch_location(index)
        with Image.open(paths[image_index]) as image:
            return image.crop(box)

    def get_patch_by_idx_(self, index):
        return self._crop(self.images, index)

    def get_hard_mask_by_idx_(self, index, mask_priority=(0, 1, 2, 3)):
        # Keep the old helper signature for existing callers; masks are class IDs.
        return self._crop(self.segmentation_masks, index)

    def __getitem__(self, index):
        image = self.get_patch_by_idx_(index)
        if image.mode not in ('L', 'RGB'):
            raise ValueError(f'Expected 8-bit L or RGB image, got {image.mode}')
        pixels = np.array(image, dtype=np.float32, copy=True)
        if pixels.ndim == 2:
            pixels = pixels[None, :, :]
        else:
            pixels = pixels.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(pixels.copy()).div_(127.5).sub_(1)
        mask = np.array(self.get_hard_mask_by_idx_(index), copy=True)
        if mask.ndim != 2:
            raise ValueError('Masks must be single-channel class-ID images')
        unknown = np.setdiff1d(np.unique(mask), self.class_values)
        if unknown.size:
            raise ValueError(f'Unknown mask class IDs: {unknown.tolist()}')
        labels = np.zeros(mask.shape, dtype=np.int64)
        for label, value in enumerate(self.class_values):
            labels[mask == value] = label
        return image_tensor, torch.from_numpy(labels)

    def __len__(self):
        return self.total_patches_amount
