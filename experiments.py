"""Shared training, evaluation and plotting for patch_training_run.ipynb."""
import random

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.PatchDataset import IGNORE_INDEX, NUM_CLASSES, PatchDataset
from models.BasicCNN import BasicCNN
from models.BasicKmeans import BasicKMeans
from models.BasicUnet import BasicUNet

COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(target, prediction):
    valid = target != IGNORE_INDEX
    return torch.bincount((target[valid] * NUM_CLASSES + prediction[valid]).cpu(),
                          minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(confusion):
    confusion = confusion.double()
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    per_class = torch.where(union > 0, confusion.diag() / union,
                            torch.full_like(union, float('nan')))
    return per_class.numpy(), float(per_class.nanmean())


def evaluate(model, loader, device):
    if isinstance(model, torch.nn.Module):
        model.eval()
    total_loss, pixels = 0.0, 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            logits = model(images)
            total_loss += F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX,
                                          reduction='sum').item()
            pixels += (target != IGNORE_INDEX).sum().item()
            confusion += confusion_matrix(target, logits.argmax(1))
    if not pixels:
        raise ValueError('Evaluation dataset contains no labelled pixels')
    per_class, mean_iou = iou_from_confusion(confusion)
    return {'loss': total_loss / pixels, 'mean_iou': mean_iou, 'iou': per_class}


def plot_predictions(model, dataset, class_names, device, title, count=4, seed=42):
    if not len(dataset):
        raise ValueError('No validation patches to plot')
    indices = np.random.default_rng(seed).choice(len(dataset), min(count, len(dataset)),
                                                replace=False)
    cmap = ListedColormap(COLORS)
    cmap.set_bad('#cccccc')
    norm = BoundaryNorm(np.arange(NUM_CLASSES + 1) - 0.5, NUM_CLASSES)
    figure, axes = plt.subplots(len(indices), 3, figsize=(12, 3 * len(indices)), squeeze=False)
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.no_grad():
        for row, index in enumerate(indices):
            image, target = dataset[int(index)]
            prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            target = target.numpy()
            invalid = target == IGNORE_INDEX
            axes[row, 0].imshow(np.ma.array(image[0].numpy(), mask=invalid),
                                cmap='gray', vmin=0, vmax=1)
            axes[row, 1].imshow(np.ma.array(target, mask=invalid), cmap=cmap, norm=norm)
            axes[row, 2].imshow(np.ma.array(prediction, mask=invalid), cmap=cmap, norm=norm)
            for column, name in enumerate(('Raw image', 'Ground truth', 'Prediction')):
                axes[row, column].set_title(f'{name} — patch {index}')
                axes[row, column].axis('off')
    figure.suptitle(title)
    figure.legend(handles=[Patch(color=COLORS[i], label=f'{i}: folder {name}')
                           for i, name in enumerate(class_names)], loc='lower center', ncol=4)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.show()
    plt.close(figure)


def _sample_training_pixels(dataset, max_pixels, seed):
    if max_pixels < NUM_CLASSES:
        raise ValueError('max_pixels must be at least four')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), min(len(dataset), 128, max_pixels), replace=False)
    per_patch = max_pixels // len(indices)
    values, labels = [], []
    for index in indices:
        image, target = dataset[int(index)]
        valid = np.flatnonzero(target.numpy().ravel() != IGNORE_INDEX)
        chosen = rng.choice(valid, min(len(valid), per_patch), replace=False)
        values.append(image.numpy().ravel()[chosen])
        labels.append(target.numpy().ravel()[chosen])
    return np.concatenate(values), np.concatenate(labels)


def run_experiment(kind, patch_size, train_samples, validation_samples, class_names,
                   epochs=5, batch_size=4, learning_rate=1e-3, seed=42,
                   background_class=None, device=None, max_kmeans_pixels=200_000,
                   kmeans_temperature=0.02):
    if len(class_names) != NUM_CLASSES:
        raise ValueError('Exactly four classes are required')
    if kind not in ('kmeans', 'cnn', 'unet'):
        raise ValueError(f'Unknown model: {kind}')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('epochs, batch_size and learning_rate must be positive')
    if not train_samples or not validation_samples:
        raise ValueError('Training and validation source-image splits must be nonempty')
    if {s.image.resolve() for s in train_samples} & {s.image.resolve() for s in validation_samples}:
        raise ValueError('Training and validation source images overlap')
    seed_everything(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train = PatchDataset(train_samples, patch_size, background_class)
    validation = PatchDataset(validation_samples, patch_size, background_class)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    title = f'{kind.upper()}, patch={patch_size}'
    print(f'{title} | device={device} | train patches={len(train)} | validation patches={len(validation)}')
    history = []
    if kind == 'kmeans':
        pixels, labels = _sample_training_pixels(train, max_kmeans_pixels, seed)
        model = BasicKMeans(seed=seed, temperature=kmeans_temperature).fit(pixels, labels)
        metrics = evaluate(model, validation_loader, device)
        training_metrics = evaluate(model, train_loader, device)
        history.append({'epoch': 0, 'train_loss': training_metrics['loss'],
                        'validation_loss': metrics['loss'], 'mean_iou': metrics['mean_iou']})
        print(f'Training K-means inertia/pixel={model.inertia:.6f}; sampled pixels={len(pixels)}')
        print(f'Train CE loss={training_metrics["loss"]:.6f}; validation CE loss={metrics["loss"]:.6f}; '
              f'validation mean IoU={metrics["mean_iou"]:.4f}')
        print('CE uses softmax(-squared centroid distance / temperature), '
              f'temperature={kmeans_temperature}; cluster-to-class mapping uses training labels only.')
    else:
        model = (BasicCNN() if kind == 'cnn' else BasicUNet()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, pixels = 0.0, 0
            for images, target in train_loader:
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), target, ignore_index=IGNORE_INDEX)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                valid = (target != IGNORE_INDEX).sum().item()
                loss_sum += loss.item() * valid
                pixels += valid
            metrics = evaluate(model, validation_loader, device)
            history.append({'epoch': epoch, 'train_loss': loss_sum / pixels,
                            'validation_loss': metrics['loss'], 'mean_iou': metrics['mean_iou']})
            print(f'Epoch {epoch}/{epochs} | train loss={loss_sum / pixels:.6f} | '
                  f'validation loss={metrics["loss"]:.6f} | mean IoU={metrics["mean_iou"]:.4f}')
    for index, name in enumerate(class_names):
        print(f'Class {index} (folder {name}) IoU={metrics["iou"][index]:.4f}')
    print('IoU is aggregated over all validation pixels; absent-in-both classes are NaN and excluded from mean.')
    plot_predictions(model, validation, class_names, device, title, seed=seed)
    # Return lightweight metrics; rerunning many experiments need not retain GPU models.
    return {'model': kind, 'patch_size': patch_size, 'history': history,
            'validation_loss': metrics['loss'], 'mean_iou': metrics['mean_iou'],
            'class_iou': metrics['iou'].tolist()}
