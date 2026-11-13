"""Train a segmentation baseline using contiguous per-image spatial splits."""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset.NickelDataset import NikelDataset
from metrics import compute_iou, update_confusion
from models.BasicCNN import BasicCNN


def pair_files(images_dir, masks_dir):
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = sorted(p for p in Path(images_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise ValueError('No input images found')
    class_dirs = {}
    for directory in sorted(Path(masks_dir).iterdir()):
        if directory.is_dir() and directory.name.isdecimal():
            class_id = int(directory.name)
            if class_id in class_dirs:
                raise ValueError(f'Duplicate class directory: {directory}')
            class_dirs[class_id] = directory
    if not class_dirs or not (set(class_dirs) - {0}):
        raise ValueError('Expected binary masks in mask/<class_id>/<image_filename>')
    masks = [{class_id: directory / image.name for class_id, directory in class_dirs.items()}
             for image in images]
    missing = [str(path) for mask_set in masks for path in mask_set.values() if not path.is_file()]
    if missing:
        raise ValueError(f'Missing same-filename masks: {missing}')
    return images, masks


def run_epoch(model, loader, device, optimizer=None, num_classes=4):
    model.train(optimizer is not None)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    loss_sum, pixels = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * targets.numel()
            pixels += targets.numel()
            update_confusion(confusion, logits.detach().argmax(1), targets)
    if not pixels:
        raise ValueError('No patches were evaluated')
    mean_iou, per_class = compute_iou(confusion)
    return loss_sum / pixels, mean_iou, per_class


def save_plot(history, output):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'test'):
        axes[0].plot(epochs, [r[f'{split}_loss'] for r in history], label=split)
        axes[1].plot(epochs, [r[f'{split}_miou'] for r in history], label=split)
    axes[0].set_ylabel('Cross-entropy loss')
    axes[1].set_ylabel('Mean IoU (including background)')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / 'curves.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if config['epochs'] <= 0 or config['batch_size'] <= 0:
        parser.error('epochs and batch_size must be positive')
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)
    images, masks = pair_files(args.images, args.masks)
    datasets = {split: NikelDataset(images, masks, config['patch_height'],
                                   split=split, train_fraction=config['train_fraction'],
                                   class_priority=config.get('class_priority'))
                for split in ('train', 'test')}
    class_values = datasets['train'].class_values
    channels = datasets['train'][0][0].shape[0]
    for path in images:
        with Image.open(path) as image:
            expected = 'L' if channels == 1 else 'RGB'
            if image.mode != expected:
                raise ValueError('All input images must have the same L or RGB mode')
    if config['batch_size'] > 1:
        widths = {region[2] for region in datasets['train'].regions}
        if len(widths) > 1:
            raise ValueError('Use batch_size=1 for images of different widths')
    loaders = {split: DataLoader(ds, batch_size=config['batch_size'],
                                shuffle=(split == 'train'),
                                num_workers=config.get('num_workers', 0))
               for split, ds in datasets.items()}
    output = args.output or Path('outputs') / args.config.stem
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    # Record exact spatial boundaries independently of patch height.
    manifest = [dict(image=str(image.resolve()),
                     masks={str(c): str(path.resolve()) for c, path in mask.items()},
                     width=region[2], height=test_region[1],
                     train_y=[region[0], region[1]],
                     test_y=[test_region[0], test_region[1]])
                for image, mask, region, test_region in
                zip(images, masks, datasets['train'].regions, datasets['test'].regions)]
    (output / 'split.json').write_text(json.dumps(manifest, indent=2))
    print(f'Device: {device}; patches: ' + ', '.join(
        f'{split}={len(ds)}' for split, ds in datasets.items()), flush=True)
    model = BasicCNN(in_channels=channels, num_classes=len(class_values)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history = []
    for epoch in range(1, config['epochs'] + 1):
        row = {'epoch': epoch}
        for split in ('train', 'test'):
            loss, miou, per_class = run_epoch(
                model, loaders[split], device, optimizer if split == 'train' else None,
                num_classes=len(class_values))
            row.update({f'{split}_loss': loss, f'{split}_miou': miou})
            row.update({f'{split}_iou_{value}': score for value, score in
                        zip(class_values, per_class)})
        history.append(row)
        with (output / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_plot(history, output)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'config': config, 'in_channels': channels,
                    'class_values': class_values,
                    'class_priority': datasets['train'].class_priority}, output / 'last.pt')
        print(f"Epoch {epoch}: train loss={row['train_loss']:.4f}, "
              f"mIoU={row['train_miou']:.4f}; test loss={row['test_loss']:.4f}, "
              f"mIoU={row['test_miou']:.4f}", flush=True)


if __name__ == '__main__':
    main()
