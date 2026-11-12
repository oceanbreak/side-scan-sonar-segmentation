# Basic segmentation models

Defaults: one grayscale input channel and four output classes. Set
`in_channels=3` for RGB tensors from the current dataset loader. Neural models
accept floating-point tensors shaped `(N, C, H, W)` and return raw logits shaped
`(N, num_classes, H, W)`. Use `CrossEntropyLoss` with integer class-index masks;
use `logits.argmax(dim=1)` for predicted masks. No softmax is applied internally.

## BasicCNN

A fully convolutional baseline that keeps the original image resolution.
All 3x3 convolutions use stride 1 and padding 1.

| Stage | Operation | Output channels | Spatial size |
| --- | --- | --- | --- |
| Input | Grayscale image | 1 | H x W |
| 1 | 3x3 convolution + ReLU | 16 | H x W |
| 2 | 3x3 convolution + ReLU | 32 | H x W |
| 3 | 3x3 convolution + ReLU | 16 | H x W |
| Head | 1x1 convolution | 4 | H x W |

**9,492 trainable parameters** with defaults. No pooling, normalization,
dropout, or fully connected layers. Each output pixel has a 7x7 receptive
field, so this is a local baseline with limited spatial context.
`base_channels` changes the 16/32/16 widths to b/2b/b.

## BasicUnet

A small two-level U-Net. Each convolution block contains two 3x3 convolutions,
each followed by ReLU, with stride 1 and padding 1.

| Stage | Operation | Output channels | Spatial size |
| --- | --- | --- | --- |
| Encoder 1 | Double convolution; save skip 1 | 16 | H x W |
| Downsample 1 | 2x2 max pool | 16 | floor(H/2) x floor(W/2) |
| Encoder 2 | Double convolution; save skip 2 | 32 | floor(H/2) x floor(W/2) |
| Downsample 2 | 2x2 max pool | 32 | floor(H/4) x floor(W/4) |
| Bottleneck | Double convolution | 64 | floor(H/4) x floor(W/4) |
| Decoder 2 | Bilinear resize to skip 2, concatenate (64+32), double convolution | 32 | floor(H/2) x floor(W/2) |
| Decoder 1 | Bilinear resize to skip 1, concatenate (32+16), double convolution | 16 | H x W |
| Head | 1x1 convolution | 4 | H x W |

**118,036 trainable parameters** with defaults. Minimum image size is 4x4.
Explicit resizing to skip sizes supports odd dimensions. Skip connections
preserve fine details while the bottleneck gathers broader context.
No normalization, dropout, or pretrained weights. `base_channels` scales
encoder/bottleneck widths to b/2b/4b.

## BasicKmeans

An unsupervised NumPy baseline for finite 2D grayscale arrays:

1. For every pixel, compute local **mean, standard deviation, and median**
   over a 5x5 window. Replicate image edges for border windows.
2. Sample up to 10,000 pixels, standardize each feature using the sample mean
   and standard deviation, and initialize four cluster centers from the sample.
3. Run up to 50 Lloyd iterations with Euclidean distance; stop when the total
   center displacement is at most 1e-4. Reseed empty clusters from the sample.
4. Assign every pixel to its nearest center and return an `(H, W)` integer mask.

`fit` learns the scaler and centers, `predict` reuses them, and `fit_predict`
combines both. Seed 0 makes fitting repeatable. Configure `n_clusters`,
`window_size` (positive odd), `max_samples`, `max_iter`, `random_state`, and `tol`.
`extract_features` exposes the `(H, W, 3)` feature array.

Cluster IDs are arbitrary and are **not semantic class labels**. Constant
images may produce fewer occupied clusters than requested. New images should
use the same intensity scale. Local feature reductions run one image row at a
time and cluster assignment is batched, but the full feature array is retained
in memory. Large sonar images may still need tiling.

## Usage

```python
import numpy as np
import torch
from models import BasicCNN, BasicUnet, BasicKmeans

images = torch.randn(2, 1, 65, 97)
cnn = BasicCNN()
unet = BasicUnet()
cnn_logits = cnn(images)
unet_logits = unet(images)
mask = unet_logits.argmax(dim=1)

sonar = np.random.default_rng(0).random((65, 97))
kmeans = BasicKmeans(n_clusters=4)
clusters = kmeans.fit_predict(sonar)
```

Run checks from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Five tests cover neural output shapes and training updates for grayscale/RGB,
odd spatial dimensions, U-Net minimum dimensions, local feature values,
repeatable two-region clustering, constant images, and invalid inputs.
These are synthetic functional checks; segmentation quality has not been
measured on real sonar data. No new dependencies are required.
