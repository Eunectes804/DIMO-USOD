# DIMO-USOD

**Photometrically Conditioned Residual Compensation for Lightweight Underwater Salient Object Detection**

DIMO-USOD is a lightweight RGB-based underwater salient object detector with **1.85M parameters**, using photometric conditioning and complementary local-contextual residual compensation.

**Notice:** This work is under review for ICASSP 2027. Our original code and model weights are for academic research and exchange only; commercial use is prohibited. Third-party components retain their original licenses.

## Installation

Python 3.8+, PyTorch 2.0+, and a matching TorchVision version are required.

```bash
pip install -r requirements.txt
```

## Data and weights

USOD10K contains **7,178 training images**, 2,051 validation images, and 1,026 test images. Organize RGB images and ground-truth masks with matching filename stems:

```text
data/USOD10K/
├── USOD10K_TR/{RGB,GT}/
├── USOD10K_Val/{RGB,GT}/
└── USOD10K_TE/{RGB,GT}/
```

Place [TinyNeXt-S ImageNet weights](https://github.com/yuffeenn/TinyNeXt/tree/main/classification) at `pretrained/tinynext_s.pth`. Download the [final DIMO-USOD checkpoint](https://github.com/Eunectes804/DIMO-USOD/releases/download/v1.0.0/dimo_usod_final.pth) and place it at `checkpoints/dimo_usod_final.pth`.

## Training

```bash
python -u train.py --data-root data/USOD10K --backbone-pretrained pretrained/tinynext_s.pth --output runs/dimo --device cuda:0
```

Defaults: 50 epochs, batch size 16, gradient accumulation 2, training sizes 256/320/352, and AdamW. Fixed photometric statistics are computed automatically from clean training images.

To resume, repeat the command with `--resume`. The final EMA weights are saved as `runs/dimo/final.pth`. Use `python train.py --help` for all options.

## Testing and evaluation

```bash
python test.py --checkpoint checkpoints/dimo_usod_final.pth --input data/USOD10K/USOD10K_TE/RGB --output results/USOD10K --device cuda:0
python eval.py --pred-dir results/USOD10K/CPCS --gt-dir data/USOD10K/USOD10K_TE/GT --output results/USOD10K/metrics
```

Testing uses RGB inputs at 320x320 and saves `Raw` and `CPCS` predictions at the original image resolution. CPCS uses alpha=2.2. Evaluation reports S-measure, mean/max E-measure, weighted F-measure, and MAE with PySODMetrics.

For other datasets, change the RGB and GT paths. UFO-120 uses `TEST/lrd` for RGB inputs and `TEST/mask` for GT.

## Code structure

```text
models/          # Backbone, decoder, and complete DIMO-USOD model
train.py         # Training and resume
test.py          # Prediction export
eval.py          # Metric evaluation
dataset.py       # Data loading
augmentation.py  # Foreground-preserving background darkening
losses.py        # Training losses
utils.py         # Optimizer, EMA, and checkpoint utilities
```

## Acknowledgements

We thank [TinyNeXt](https://github.com/yuffeenn/TinyNeXt), [GAPNet](https://github.com/yuhuan-wu/GAPNet), and [AFF](https://github.com/YimianDai/open-aff). Third-party attribution and licenses are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
