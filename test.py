"""RGB-only inference: save both continuous Raw and fixed-CPCS saliency maps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from models.dimo import cpcs
from utils import load_model, seed_all, sha256, write_json

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class RGBImages(Dataset):
    def __init__(self, path, size=320):
        path = Path(path)
        self.files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        self.size = size
        if not self.files:
            raise ValueError('No RGB images found')
        if len({p.stem for p in self.files}) != len(self.files):
            raise ValueError('Duplicate image stems would overwrite predictions')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as source:
            rgb = source.convert('RGB')
            width, height = rgb.size
            rgb = TF.resize(rgb, [self.size, self.size], InterpolationMode.BILINEAR, antialias=True)
            x = TF.normalize(TF.to_tensor(rgb), [.485, .456, .406], [.229, .224, .225])
        return dict(image=x, name=path.stem, height=height, width=width)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/dimo_usod_final.pth'))
    p.add_argument('--input', type=Path, required=True, help='One RGB file or a flat RGB directory')
    p.add_argument('--output', type=Path, default=Path('results/predictions'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--alpha', type=float, default=2.2)
    p.add_argument('--save-floats', action='store_true', help='Also save native-grid float32 probabilities')
    p.add_argument('--trusted-legacy', action='store_true', help='Allow pickle loading for a trusted old checkpoint')
    a = p.parse_args()
    if a.size < 64 or a.size % 32 or a.batch_size < 1 or a.workers < 0:
        p.error('Size must be >=64 and divisible by 32; batch-size positive, workers nonnegative')
    torch.set_num_threads(4)
    seed_all(2026)
    model, checkpoint = load_model(a.checkpoint, a.device, a.trusted_legacy)
    model.return_all = False
    dataset = RGBImages(a.input, a.size)
    loader = DataLoader(dataset, batch_size=a.batch_size, num_workers=a.workers, shuffle=False)
    for name in ('Raw', 'CPCS'):
        (a.output / name).mkdir(parents=True, exist_ok=True)
    if a.save_floats:
        (a.output / 'native').mkdir(parents=True, exist_ok=True)
    seen, flips = 0, 0
    with torch.inference_mode():
        for batch in loader:
            raw = model(batch['image'].to(a.device)).float()
            shaped = cpcs(raw, a.alpha)
            if not torch.isfinite(raw).all() or not torch.isfinite(shaped).all():
                raise FloatingPointError('Nonfinite saliency map')
            flips += int(((raw >= .5) != (shaped >= .5)).sum())
            for i, name in enumerate(batch['name']):
                size = (int(batch['height'][i]), int(batch['width'][i]))
                for tag, value in [('Raw', raw), ('CPCS', shaped)]:
                    resized = F.interpolate(value[i:i+1], size=size, mode='bilinear', align_corners=False)[0, 0]
                    image = resized.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    Image.fromarray(image).save(a.output / tag / (name + '.png'))
                if a.save_floats:
                    np.savez_compressed(a.output / 'native' / (name + '.npz'), raw=raw[i, 0].cpu().numpy(), cpcs=shaped[i, 0].cpu().numpy())
            seen += len(raw)
            print('Inference {}/{}'.format(seen, len(dataset)), flush=True)
    report = dict(images=seen, checkpoint_sha256=sha256(a.checkpoint), epoch=checkpoint.get('epoch'),
                  input_size=a.size, alpha=a.alpha, native_binary_changes=flips,
                  protocol='RGB -> resize/normalize -> sigmoid at input grid -> optional CPCS -> bilinear to original resolution -> round uint8',
                  minmax_normalization_on_export=False)
    write_json(a.output / 'inference.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
