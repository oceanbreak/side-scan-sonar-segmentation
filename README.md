# side-scan-sonar-segmentation
Researching different models for segmenting SSS images

## Data under study:
Large-scale SSS images (1 channel),
binary segmentation masks for each class

## Result:
Segmented SSS image of the same size
## Training CNN and U-Net

Place images and **single-channel class-index masks** in separate directories.
Names must match by relative filename stem (for example `images/scan1.jpg` and
`masks/scan1.png`). Masks should use lossless formats. Separate binary masks
per class must first be combined into one class-index image. Defaults map raw
mask values `0, 1, 7, 10` to targets `0, 1, 2, 3`; unknown values raise an error.
Images are converted to 8-bit grayscale and normalized to [-1, 1].

```bash
source .venv/bin/activate
python train.py --images /path/to/images --masks /path/to/masks \
  --model unet --epochs 20 --batch-size 4 --resize 256 256 \
  --plot --output-dir /workspace/runs/unet
```

Use `--model cnn` for the convolutional baseline. Training uses Adam and
cross-entropy with automatic CUDA/CPU selection (`--device cpu` overrides it).
`tqdm` displays training/validation progress and running loss for every epoch.
`--plot` updates `loss.png` after every epoch without needing a display;
`--live-plot` additionally updates a Matplotlib window on an interactive backend.
In notebooks use an interactive Matplotlib backend if you want a live window.
The plot is refreshed once per epoch, not per batch.

The output directory contains `last.pt`, `best.pt`, `history.json`, `config.json`,
and optionally `loss.png`. The best checkpoint uses validation loss (training
loss if validation is disabled). Checkpoints contain model/optimizer state,
epoch, model class, and loss history. To reload weights, instantiate the same
model/class count from `config.json`, then load `checkpoint['model_state_dict']`.
Automatic resumption is not implemented.

By default, 20% of source images (at least one) are used for validation; the
seeded split happens **before patch extraction**. Use `--val-fraction 0` for
training without validation, including a dataset with only one image.
`--patch-height 256` crops full-width horizontal strips before resizing;
incomplete bottom strips are discarded. Without it, whole images are resized.
CLI resizing defaults to 256x256 so batches have uniform dimensions. Choose a
suitable resolution/aspect ratio for your sonar data. U-Net needs dimensions >=4.
Use `python train.py --help` for all options.

### Python / notebook interface

```python
from torch.utils.data import DataLoader
from dataset.SegmentationDataset import SegmentationDataset, discover_pairs
from models import BasicCNN
from train import fit

pairs = discover_pairs('/path/to/images', '/path/to/masks')
dataset = SegmentationDataset(pairs, resize=(256, 256))
loader = DataLoader(dataset, batch_size=4, shuffle=True)
history = fit(BasicCNN(), loader, epochs=10,
              output_dir='/workspace/runs/cnn', plot=True)
```

Pass a separate `val_loader` to `fit` for validation. Dataset loading requires
Pillow, NumPy and PyTorch; progress/plots use tqdm and Matplotlib, already in the
development environment. The new `SegmentationDataset` supports grayscale and
correct patch indexing; the older `NikelDataset` remains unchanged.
K-means uses its independent `fit`/`predict` API rather than this gradient-based
trainer. See [model structures](models/README.md).

Run functional tests with:

```bash
python -m unittest discover -s tests -v
```
