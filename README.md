# side-scan-sonar-segmentation
Researching different models for segmenting SSS images

## Data under study:
Large-scale SSS images (1 channel),
binary segmentation masks for each class

## Result:
Segmented SSS image of the same size
## Training runs

Use Python 3.9+ with PyTorch, NumPy, Pillow, and Matplotlib installed. In the
prepared cloud environment, activate `source .venv/bin/activate`.
For a fresh environment, install a suitable CPU/CUDA PyTorch build first, then
run `python -m pip install -r requirements.txt`.

Place sonar images and their per-class binary masks in this structure:

```text
path/
  img/01.png
  img/02.png
  mask/1/01.png
  mask/1/02.png
  mask/7/01.png
  mask/7/02.png
  mask/10/01.png
  mask/10/02.png
```

Class IDs are discovered from numeric subdirectory names under `mask/`; any
nonnegative integer class IDs are supported. Every image must have a mask with
exactly the same filename in every discovered class directory, including an
all-zero mask when that class is absent. Missing files and dimension mismatches
raise errors rather than silently treating missing annotations as background.

Images can be 8-bit grayscale (`L`) or RGB, with the same channel mode throughout
a run. Masks must be single-channel binary images (0/1 or 0/255), stored losslessly
and matching source dimensions. Nonzero pixels mark membership in that class.
Pixels outside all foreground masks become background (class 0); an optional
`mask/0/` directory does not override foreground classes. Class IDs are mapped to
contiguous training labels, and the original IDs are retained in CSV/checkpoints.

Masks are cropped at identical Y coordinates, then combined **per patch**.
By default, the highest numeric foreground class wins where masks overlap:
for classes 1, 7, 10, priority is 10 > 7 > 1. To override, add `"class_priority":
[10, 7, 1]` to a run JSON: list every foreground class once, **lowest to highest**
priority (the last entry wins). This example makes class 1 win. No source files
are modified or combined on disk. Direct Python callers can still supply legacy
combined class-ID mask paths for classes 0, 1, 7, 10 to `NikelDataset`.

Run each experiment from the repository root:

```bash
python train.py --config runs/patch_64.json --images /path/img --masks /path/mask
python train.py --config runs/patch_128.json --images /path/img --masks /path/mask
python train.py --config runs/patch_256.json --images /path/img --masks /path/mask
```

The three run files differ only in patch height (64, 128, 256). All use the same
seed, a small `BasicCNN` segmentation baseline, Adam, 20 epochs, and batch size 1.
Use `--epochs 1` for a smoke run, `--device cpu` to force CPU, or `--output PATH`
for a custom **new** output directory. Existing output directories are rejected
to prevent accidental overwriting. CUDA is selected automatically when available.
Batch size 1 supports different source widths; larger batches require equal widths.

### Spatial split and patches

For **each image and its mask**, the top `floor(height * 0.8)` rows are training
and the remaining bottom rows are test. Change `train_fraction` in a run file
if needed. This boundary is independent of patch height, allowing comparisons
against the same held-out spatial region. Splitting precedes patching:

- Patches preserve the full image width; cropping happens along Y only.
- Image and mask use identical crop coordinates, without resizing.
- Only complete patches are used; leftover rows at the end of each region are dropped.
- Every source must have at least one complete patch in both regions. For 256-row
  patches, an image needs about 1280 rows or more.
- Dataset indexing now returns patches; the default height is 128.

No pixels overlap across train/test. Nearby regions of the same acquisition can
still be correlated; this measures spatial holdout performance within images,
not generalization to entirely unseen surveys. Test metrics are reported each
epoch, but do not select checkpoints or update model weights. Reserve a separate
validation split if using these curves to tune hyperparameters.

### Outputs and metrics

Each run creates `outputs/patch_<height>/` containing:

- `curves.png`: train/test cross-entropy loss and mean IoU versus epoch.
- `history.csv`: loss, mean IoU, and IoU for each original class ID, per epoch.
- `last.pt`: final model/optimizer state, epoch, class mapping, overlap priority, and configuration.
- `config.json` and `split.json`: run settings, source paths, and exact Y boundaries.

IoU is intersection / union, accumulated over **all pixels in the epoch**, then
averaged across classes with nonzero union, including background. Classes absent
from both predictions and targets have undefined IoU (`nan` in CSV) and are
excluded from the mean. Loss is averaged by pixel count. Train metrics are
measured during optimization; test metrics use evaluation mode without gradients.
Large compressed images may still need to be decoded in full by Pillow when
cropped; smaller tensors reduce training memory but do not eliminate decode costs.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```
