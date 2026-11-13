"""Full-width sonar patches, optionally restricted to a spatial train/test split."""
import operator
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NikelDataset(Dataset):
    """Cut along Y only. Drop incomplete patches at each region's bottom edge.

    ``split=None`` uses whole images. Otherwise the top ``train_fraction`` of
    every image is train and the remaining bottom region is test. Splitting is
    done before patching, so no patch crosses the train/test boundary.
    Masks are dictionaries {class_id: binary_mask_path}, one per image.
    Nonzero pixels indicate membership; uncovered pixels are background (0).
    Legacy combined class-ID masks are also accepted.
    """

    class_values = (0, 1, 7, 10)

    def __init__(self, images, segmentation_masks, patch_height=128,
                 split=None, train_fraction=0.8, class_priority=None):
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
            raise ValueError('Each image needs one matching mask set')
        self.binary_masks = bool(self.segmentation_masks and
                                 isinstance(self.segmentation_masks[0], Mapping))
        if any(isinstance(masks, Mapping) != self.binary_masks
               for masks in self.segmentation_masks):
            raise ValueError('Cannot mix binary mask sets and combined class-ID masks')
        if self.binary_masks:
            classes = set(self.segmentation_masks[0])
            if not classes or any(type(c) is not int or c < 0 for c in classes):
                raise ValueError('Mask class IDs must be nonnegative integers')
            if any(set(masks) != classes for masks in self.segmentation_masks):
                raise ValueError('Every image must have the same mask classes')
            self.class_values = (0, *sorted(classes - {0}))
            foreground = set(self.class_values) - {0}
            self.class_priority = tuple(sorted(foreground) if class_priority is None
                                        else class_priority)
            if (set(self.class_priority) != foreground or
                    len(self.class_priority) != len(foreground)):
                raise ValueError('class_priority must list each foreground class once')
        self.patch_height = patch_height
        self.split = split
        self.train_fraction = train_fraction
        self.regions = []
        self.index_intervals = [0]
        for image_file, mask_file in zip(self.images, self.segmentation_masks):
            with Image.open(image_file) as image:
                width, height = image.size
            paths = mask_file.values() if self.binary_masks else [mask_file]
            for path in paths:
                with Image.open(path) as mask:
                    if mask.size != (width, height):
                        raise ValueError(f'Image/mask size mismatch: {image_file}, {path}')
                    if self.binary_masks and mask.mode not in ('1', 'L', 'P', 'I', 'I;16'):
                        raise ValueError(f'Binary masks must be single-channel: {path}')
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

    def get_hard_mask_by_idx_(self, index, mask_priority=None):
        if not self.binary_masks:
            return self._crop(self.segmentation_masks, index)
        source, box = self.patch_location(index)
        priority = self.class_priority if mask_priority is None else tuple(mask_priority)
        if set(priority) != set(self.class_values) - {0} or len(priority) != len(self.class_values) - 1:
            raise ValueError('mask_priority must list each foreground class once')
        labels = np.zeros((box[3] - box[1], box[2]), dtype=np.int32)
        # Later classes win on overlap. Crop before combining to avoid full-size arrays.
        for class_id in priority:
            with Image.open(self.segmentation_masks[source][class_id]) as mask:
                membership = np.array(mask.crop(box), copy=True)
            if not np.isin(membership, (0, 1, 255)).all():
                raise ValueError('Binary masks must use 0/1 or 0/255 values')
            labels[membership != 0] = class_id
        return Image.fromarray(labels)

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
