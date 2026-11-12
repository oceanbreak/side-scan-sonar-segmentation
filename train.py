"""Simple CNN/U-Net training API and command-line interface."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.SegmentationDataset import SegmentationDataset, discover_pairs
from models import BasicCNN, BasicUnet


def fit(model, train_loader, val_loader=None, epochs=10, lr=1e-3,
        device=None, output_dir='runs/basic', plot=False, live_plot=False,
        progress=True):
    """Train with Adam/cross entropy; save last/best checkpoints and loss history.

    Loaders must yield (float NCHW images, long NHW class-index masks).
    live_plot uses the current Matplotlib backend; plot alone works headlessly.
    """
    if epochs < 1 or lr <= 0 or len(train_loader) == 0:
        raise ValueError('Need positive epochs/lr and a nonempty training loader')
    if val_loader is not None and len(val_loader) == 0:
        raise ValueError('Validation loader must be nonempty')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = {'train_loss': [], 'val_loss': []}
    best = float('inf')
    figure = None
    if plot or live_plot:
        import matplotlib
        if not live_plot:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if live_plot:
            plt.ion()
        figure, axis = plt.subplots()
        if live_plot:
            plt.show(block=False)
    try:
        for epoch in range(1, epochs + 1):
            for phase, loader in (('train', train_loader), ('val', val_loader)):
                if loader is None:
                    continue
                training = phase == 'train'
                model.train(training)
                total_loss, count = 0.0, 0
                bar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} {phase}', disable=not progress)
                with bar, torch.set_grad_enabled(training):
                    for images, targets in bar:
                        images, targets = images.to(device), targets.to(device)
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                        logits = model(images)
                        loss = criterion(logits, targets)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite loss; check data and learning rate')
                        if training:
                            loss.backward()
                            optimizer.step()
                        pixels = targets.numel()
                        total_loss += loss.item() * pixels
                        count += pixels
                        bar.set_postfix(loss=f'{total_loss / count:.4f}')
                if count == 0:
                    raise ValueError('Loader yielded no target pixels')
                history[phase + '_loss'].append(total_loss / count)
            score = history['val_loss'][-1] if val_loader is not None else history['train_loss'][-1]
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(), 'history': history,
                          'model_class': type(model).__name__}
            torch.save(checkpoint, output / 'last.pt')
            if score < best:
                best = score
                torch.save(checkpoint, output / 'best.pt')
            (output / 'history.json').write_text(json.dumps(history, indent=2))
            if figure is not None:
                axis.clear()
                for name, values in history.items():
                    if values:
                        axis.plot(range(1, len(values) + 1), values, label=name)
                axis.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training progress')
                axis.legend()
                figure.tight_layout()
                figure.savefig(output / 'loss.png')
                if live_plot:
                    figure.canvas.draw_idle()
                    plt.pause(0.01)
    finally:
        if figure is not None:
            plt.close(figure)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--model', choices=('cnn', 'unet'), default='cnn')
    parser.add_argument('--class-values', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Raw values for class-index masks; unused for class-folder binary masks')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch-height', type=int)
    parser.add_argument('--resize', nargs=2, type=int, default=[256, 256], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default='runs/basic')
    parser.add_argument('--plot', action='store_true', help='Update loss.png after each epoch')
    parser.add_argument('--live-plot', action='store_true', help='Also refresh a Matplotlib window')
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        parser.error('--val-fraction must be in [0, 1)')
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('Need positive batch size and nonnegative worker count')
    torch.manual_seed(args.seed)
    pairs = discover_pairs(args.images, args.masks)
    # Split source images BEFORE patch extraction to avoid patch leakage.
    order = torch.randperm(len(pairs)).tolist()
    pairs = [pairs[i] for i in order]
    n_val = max(1, int(len(pairs) * args.val_fraction)) if args.val_fraction else 0
    if n_val >= len(pairs):
        parser.error('Validation needs at least two image pairs; use --val-fraction 0 for one')
    options = dict(class_values=args.class_values, patch_height=args.patch_height, resize=args.resize)
    train_data = SegmentationDataset(pairs[n_val:], **options)
    val_data = SegmentationDataset(pairs[:n_val], **options) if n_val else None
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, **loader_options) if val_data is not None else None
    model = (BasicCNN if args.model == 'cnn' else BasicUnet)(num_classes=train_data.num_classes)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), class_names=train_data.class_names,
                  num_classes=train_data.num_classes, binary_masks=train_data.binary_masks)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'{type(model).__name__}: {len(train_data)} training patches, '
          f'{len(val_data) if val_data is not None else 0} validation patches')
    fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        device=args.device, output_dir=output, plot=args.plot, live_plot=args.live_plot)
    print(f'Artifacts saved to {output}')


if __name__ == '__main__':
    main()
