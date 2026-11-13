"""Epoch-level multiclass IoU from a confusion matrix (rows are targets)."""
import torch


def update_confusion(confusion, predictions, targets):
    classes = confusion.shape[0]
    counts = torch.bincount(
        (targets.reshape(-1) * classes + predictions.reshape(-1)),
        minlength=classes * classes,
    )
    confusion += counts.reshape(classes, classes)


def compute_iou(confusion):
    confusion = confusion.to(torch.float64)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    per_class = torch.full_like(union, float('nan'))
    per_class[present] = intersection[present] / union[present]
    mean = per_class[present].mean().item() if present.any() else float('nan')
    return mean, per_class.tolist()
