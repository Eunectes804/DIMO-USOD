"""USOD10K RGB/GT loader and original GAPNet granularity targets."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


cv2.setNumThreads(1)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_granularity_targets(mask: torch.Tensor) -> torch.Tensor:
    """Build GAPNet labels: full, boundary, center, center+other,
    boundary+other, and other.
    """

    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"mask必须为[1,H,W]，当前为{tuple(mask.shape)}")
    binary = mask[0].detach().cpu().numpy().astype(np.uint8)
    result = np.zeros((6, *binary.shape), dtype=np.float32)
    result[0] = binary
    if binary.max() == 0:
        return torch.from_numpy(result)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, maskSize=5)
    foreground_distances = distance[distance > 0]
    if foreground_distances.size == 0:
        return torch.from_numpy(result)

    partition_index = int(foreground_distances.size * 0.8)
    center_threshold = np.partition(
        foreground_distances, partition_index
    )[partition_index]
    if center_threshold < 5:
        return torch.from_numpy(result)

    center = distance > center_threshold
    boundary = (distance > 0) & (distance < 5)
    foreground = binary.astype(bool)
    other = foreground & ~(center | boundary)
    result[1] = boundary
    result[2] = center
    result[3] = center | other
    result[4] = boundary | other
    result[5] = other
    return torch.from_numpy(result)




class USOD10KDataset(Dataset):
    SPLIT_DIRECTORIES = {
        "train": "USOD10K_TR",
        "val": "USOD10K_Val",
        "test": "USOD10K_TE",
    }

    def __init__(
        self,
        data_root,
        split: str,
        image_size: int,
        augment: bool = False,
        return_granularity_targets: bool = False,
        image_dir_override=None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.image_size = int(image_size)
        self.augment = bool(augment and self.split == "train")
        self.return_granularity_targets = return_granularity_targets
        if self.split not in self.SPLIT_DIRECTORIES:
            raise ValueError("split must be train, val or test")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        split_root = self.data_root / self.SPLIT_DIRECTORIES[self.split]
        self.image_dir = (
            Path(image_dir_override).resolve()
            if image_dir_override is not None
            else split_root / "RGB"
        )
        self.mask_dir = split_root / "GT"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directories missing: {self.image_dir} / {self.mask_dir}"
            )

        image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        masks: Dict[str, Path] = {
            path.stem: path for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        self.samples: List[Tuple[Path, Path]] = [
            (path, masks[path.stem])
            for path in image_paths if path.stem in masks
        ]
        if not self.samples:
            raise RuntimeError(f"No paired RGB/GT files: {split_root}")
        print(
            f"USOD10K {self.split}: paired {len(self.samples)} samples"
        )
        if image_dir_override is not None:
            print(f"USOD10K {self.split}: using external RGB directory {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _synchronized_augmentation(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Apply the paper's paired crop-resize and vertical flip.
        crop_pixels = int(7.0 / 224.0 * self.image_size)
        if torch.rand(()).item() < 0.5 and crop_pixels > 0:
            crop_x = int(torch.randint(0, crop_pixels + 1, ()).item())
            crop_y = int(torch.randint(0, crop_pixels + 1, ()).item())
            width = self.image_size - 2 * crop_x
            height = self.image_size - 2 * crop_y
            if width > 0 and height > 0:
                image = TF.resized_crop(
                    image, crop_y, crop_x, height, width,
                    [self.image_size, self.image_size],
                    InterpolationMode.BILINEAR, antialias=True,
                )
                mask = TF.resized_crop(
                    mask, crop_y, crop_x, height, width,
                    [self.image_size, self.image_size],
                    InterpolationMode.NEAREST,
                )
        if torch.rand(()).item() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            InterpolationMode.NEAREST,
        )
        if self.augment:
            image, mask = self._synchronized_augmentation(image, mask)

        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask = (TF.to_tensor(mask) >= 0.5).float()
        sample = {"image": image, "mask": mask, "name": image_path.stem}
        if self.return_granularity_targets:
            sample["granularity_mask"] = build_granularity_targets(mask)
        return sample

def seed_worker(_):
    import random
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def make_loader(root, split, size, batch_size, workers, seed, limit=0):
    from torch.utils.data import DataLoader, Subset
    ds = USOD10KDataset(root, split, size, augment=split == 'train',
                       return_granularity_targets=split == 'train')
    ids = list(range(len(ds)))
    if limit and limit < len(ids):
        ids = sorted(np.random.RandomState(seed + (0 if split == 'train' else 100000)).choice(ids, limit, replace=False).tolist())
    names = [ds.samples[i][0].stem for i in ids]
    generator = torch.Generator().manual_seed(seed + (size if split == 'train' else 100000))
    loader = DataLoader(Subset(ds, ids), batch_size=batch_size, shuffle=split == 'train',
                        drop_last=split == 'train', num_workers=workers, pin_memory=True,
                        persistent_workers=False, generator=generator, worker_init_fn=seed_worker)
    if not len(loader):
        raise ValueError('No batches; training needs at least one physical batch')
    return loader, generator, names
