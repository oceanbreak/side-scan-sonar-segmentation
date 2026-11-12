# side-scan-sonar-segmentation
Researching different models for segmenting SSS images

## Data under study:
Large-scale SSS images (1 channel),
binary segmentation masks for each class

## Result:
Segmented SSS image of the same size
## Training CNN and U-Net

The loader automatically supports your per-class binary mask layout:

```text
IMAGE/
  01.png
  02.png
MASKS/
  1/
    01.png
    02.png
  2/
    01.png
    02.png
  ...
```

Each class folder must contain a mask matching every source image by filename
stem (extensions may differ). Missing/extra masks, duplicate stems, or mismatched
image/mask dimensions raise an error. Use lossless mask formats such as PNG.
White/nonzero pixels belong to the class; black pixels do not. RGB binary masks
are converted to grayscale. Use all-black masks when a class is absent.

Background is label **0**. Class folders sort numerically if all names are
numeric, otherwise alphabetically, and map to labels **1..N**. For example,
folders `1`, `2`, `10` map to labels `1`, `2`, `3`. If masks overlap, the later
folder in this order wins. The model automatically gets **N+1 output channels**.
`config.json` saves `class_names` (in output-channel order) and `num_classes`.
This creates one exclusive class per pixel, not multilabel predictions.

Flat single-channel class-index masks are still supported: filenames match
source stems, and `--class-values 1 2 3 4` maps raw values to consecutive labels.
That option is ignored for class-folder masks. Images are converted to 8-bit
grayscale and normalized to [-1, 1].

```bash
source .venv/bin/activate
python train.py --images /path/to/IMAGE --masks /path/to/MASKS \
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
