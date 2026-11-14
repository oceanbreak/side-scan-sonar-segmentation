# side-scan-sonar-segmentation
Researching different models for segmenting SSS images

## Data under study:
Large-scale SSS images (1 channel),
binary segmentation masks for each class

## Result:
Segmented SSS image of the same size

## Patch training notebook

Open **[patch_training_run.ipynb](patch_training_run.ipynb)** with the repository's
Python environment. The notebook provides separate runnable cells for K-means,
CNN at 128 and 256 pixels, and U-Net at 128 and 256 pixels. Edit patch sizes,
epochs, batch size and `DATA_ROOT` in the configuration cell. Each experiment
prints training/validation loss, validation mean and per-class IoU, and plots up
to four validation patches with raw image, ground truth and predicted classes.

Use Python 3.9+ and install `requirements.txt`. For CPU-only setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter lab patch_training_run.ipynb
```

The notebook expects `DATA_ROOT/image/<name>.<ext>` and
`DATA_ROOT/mask/<class number>/<name>.<ext>` for each of exactly four numeric
class folders. Filename stems must match; extensions may differ. Numeric folder
names are sorted and mapped to IDs 0–3, including noncontiguous numbers such as
0, 1, 7, 10. Masks may be single-channel or RGB/RGBA PNGs: black means absent,
white (or any nonzero color channel) marks a class. Alpha is ignored. All four
masks must exist for every source image. Overlaps are rejected.
Uncovered pixels raise an error unless `BACKGROUND_CLASS` explicitly selects
one of the four class IDs. Input images must be 8-bit; grayscale is supported
and RGB inputs are converted to grayscale.

At least two source images are needed. Source images are split before tiling;
validation images are shared across all experiments. Partial border patches
are padded and padding is excluded from metrics. Classes absent in both truth
and prediction are excluded from mean IoU. K-means fits intensity clusters and
learns cluster-to-class assignments on training pixels only; its reported
cross-entropy uses distance-derived probabilities, while inertia is reported
separately as its training objective. Patch size does not add spatial features
to the K-means baseline.

The notebook uses `dataset/PatchDataset.py` and the baseline implementations in
`models/`; the older `NikelDataset` remains available unchanged. No data or
trained weights are included. Local `data/` is ignored by Git. Larger patches
may require a smaller batch size on GPU. Results depend on the supplied data;
synthetic smoke validation does not establish sonar segmentation quality.

Run the tests, including all five notebook experiment cells on temporary
synthetic data:

```bash
python -m unittest discover -s tests -v
```
