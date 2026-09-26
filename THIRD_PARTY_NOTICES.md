# Third-party notices

## TinyNeXt

`models/backbone.py` adapts the TinyNeXt-S blocks and configuration for dense feature extraction. Original project: https://github.com/yuffeenn/TinyNeXt . The upstream MIT license is reproduced in full below.

```text
The MIT License (MIT)
Copyright (c) 2025 Fanhong Zeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GAPNet

The decoder, global/cross-scale attention, granular pyramid convolutions, and granularity supervision follow GAPNet. Adapted operators are in `models/decoder.py`; model assembly is in `models/dimo.py`; targets and losses are in `dataset.py` and `losses.py`.

Original project: https://github.com/yuhuan-wu/GAPNet . Paper: *GAPNet: A Lightweight Framework for Image and Video Salient Object Detection via Granularity-Aware Paradigm*.

The upstream README states academic research use and directs commercial-use inquiries to the authors. This repository does not grant additional rights to upstream GAPNet material.

## Attentional Feature Fusion

The complementary local/global gate design is informed by AFF: https://github.com/YimianDai/open-aff . DIMO applies allocation to signed residual corrections using explicit photometric conditions. AFF is a methodological reference; its source code is not bundled here.

## Dependencies and original additions

PyTorch, TorchVision, NumPy, Pillow, OpenCV, SciPy, and PySODMetrics retain their own licenses and are installed as dependencies. Dataset images and third-party checkpoints are not included in the source distribution.

Original DIMO-USOD code and model weights are provided for non-commercial academic research and exchange only, as stated in the README. Third-party components retain their original licenses.
