"""Foreground-preserving background darkening for training."""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple
import torch
from torch.nn import functional as F

def _translate_integer(
    image: torch.Tensor,
    dx: int,
    dy: int,
) -> torch.Tensor:
    """Translate a single-channel mask without wraparound."""

    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("_translate_integer expects [1,1,H,W]")
    _, _, height, width = image.shape
    dx = max(-width, min(width, int(dx)))
    dy = max(-height, min(height, int(dy)))
    result = torch.zeros_like(image)

    source_x0 = max(-dx, 0)
    source_x1 = min(width - dx, width)
    source_y0 = max(-dy, 0)
    source_y1 = min(height - dy, height)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return result
    destination_x0 = source_x0 + dx
    destination_x1 = source_x1 + dx
    destination_y0 = source_y0 + dy
    destination_y1 = source_y1 + dy
    result[
        :, :, destination_y0:destination_y1, destination_x0:destination_x1
    ] = image[:, :, source_y0:source_y1, source_x0:source_x1]
    return result


def _gaussian_kernel_1d(
    sigma: float,
    maximum_radius: int,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    if sigma <= 0 or maximum_radius <= 0:
        return reference.new_ones(1), 0
    radius = min(max(int(math.ceil(3.0 * sigma)), 1), maximum_radius)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=reference.device,
        dtype=reference.dtype,
    )
    kernel = torch.exp(-0.5 * (coordinates / max(sigma, 1e-4)).square())
    return kernel / kernel.sum().clamp_min(1e-8), radius


def _gaussian_blur(
    image: torch.Tensor,
    sigma_y: float,
    sigma_x: Optional[float] = None,
) -> torch.Tensor:
    """Apply separable Gaussian blur to a single-channel mask."""

    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("_gaussian_blur expects [N,1,H,W]")
    sigma_x = sigma_y if sigma_x is None else sigma_x
    height, width = image.shape[-2:]
    kernel_x, radius_x = _gaussian_kernel_1d(
        float(sigma_x), max(width - 1, 0), image
    )
    kernel_y, radius_y = _gaussian_kernel_1d(
        float(sigma_y), max(height - 1, 0), image
    )
    result = image
    if radius_x:
        result = F.pad(result, (radius_x, radius_x, 0, 0), mode="replicate")
        result = F.conv2d(result, kernel_x.view(1, 1, 1, -1))
    if radius_y:
        result = F.pad(result, (0, 0, radius_y, radius_y), mode="replicate")
        result = F.conv2d(result, kernel_y.view(1, 1, -1, 1))
    return result


def _elliptical_dilate(
    binary: torch.Tensor,
    radius_y: int,
    radius_x: int,
) -> torch.Tensor:
    """Dilate a binary mask with an elliptical footprint."""

    radius_y = max(int(radius_y), 1)
    radius_x = max(int(radius_x), 1)
    yy = torch.arange(
        -radius_y,
        radius_y + 1,
        device=binary.device,
        dtype=binary.dtype,
    ).view(-1, 1)
    xx = torch.arange(
        -radius_x,
        radius_x + 1,
        device=binary.device,
        dtype=binary.dtype,
    ).view(1, -1)
    footprint = (
        (yy / float(radius_y)).square()
        + (xx / float(radius_x)).square()
        <= 1.0
    ).to(binary.dtype)
    dilated = F.conv2d(
        binary,
        footprint.view(1, 1, *footprint.shape),
        padding=(radius_y, radius_x),
    )
    return (dilated > 0).to(binary.dtype)

class BackgroundMaskGenerator:
    def __init__(self):
        self.family_probabilities = (0.50, 0.30, 0.20)
        self.epsilon = 1e-6
        self.target_mask_mean = 0.060
        self.maximum_shadow_to_foreground_ratio = 1.0

    @staticmethod
    def _validate_inputs(
        normalized_images: torch.Tensor,
        full_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = normalized_images.float()
        target = full_target.float().clamp(0.0, 1.0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "normalized_images must be [B,3,H,W], got "
                f"{tuple(images.shape)}"
            )
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(
                f"full_target must be [B,1,H,W], got {tuple(target.shape)}"
            )
        if images.shape[0] != target.shape[0] or images.shape[-2:] != target.shape[-2:]:
            raise ValueError("image and target shapes do not align")
        if images.shape[0] == 0:
            raise ValueError("Cannot darken an empty batch")
        if not torch.isfinite(images).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite image/target values")
        return images, target

    @staticmethod
    def _random(
        shape: Sequence[int],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        # Sample geometry on CPU from the seeded generator.
        return torch.rand(tuple(shape), generator=generator, device="cpu")

    def _sample_families(
        self,
        count: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        uniforms = self._random((count,), generator)
        boundaries = torch.tensor(self.family_probabilities).cumsum(0)
        return torch.bucketize(uniforms, boundaries[:-1], right=False).long()

    def _raw_contact(
        self,
        guide: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        height, width = guide.shape[-2:]
        horizontal_sign = -1.0 if float(parameters[0]) < 0.5 else 1.0
        maximum_dx = horizontal_sign * (0.012 + 0.024 * float(parameters[1])) * width
        maximum_dy = (0.055 + 0.035 * float(parameters[2])) * height
        raw = torch.zeros_like(guide)
        for fraction in torch.linspace(0.10, 1.0, 10).tolist():
            shifted = _translate_integer(
                guide,
                round(maximum_dx * fraction),
                round(maximum_dy * fraction),
            )
            raw = torch.maximum(raw, shifted * (1.0 - 0.20 * fraction))
        sigma = max(1.2, (0.009 + 0.005 * float(parameters[3])) * min(height, width))
        return _gaussian_blur(raw, sigma)

    def _raw_lower_wrap(
        self,
        guide: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        height, width = guide.shape[-2:]
        scale = min(height, width)
        radius_x = max(round((0.045 + 0.030 * float(parameters[0])) * scale), 2)
        radius_y = max(round((0.025 + 0.018 * float(parameters[1])) * scale), 1)
        ring = (_elliptical_dilate(guide, radius_y, radius_x) - guide).clamp_min(0.0)

        yy = torch.arange(height, device=guide.device, dtype=guide.dtype).view(
            1, 1, height, 1
        )
        foreground_mass = guide.sum().clamp_min(1.0)
        center_y = (guide * yy).sum() / foreground_mass
        gate_width = max((0.015 + 0.008 * float(parameters[2])) * height, 1.0)
        lower_gate = torch.sigmoid((yy - center_y) / gate_width)
        ring = ring * lower_gate

        horizontal_sign = -1.0 if float(parameters[3]) < 0.5 else 1.0
        maximum_dx = horizontal_sign * (0.030 + 0.040 * float(parameters[4])) * width
        maximum_dy = (0.055 + 0.035 * float(parameters[5])) * height
        raw = ring.clone()
        for fraction in torch.linspace(0.15, 1.0, 8).tolist():
            shifted = _translate_integer(
                ring,
                round(maximum_dx * fraction),
                round(maximum_dy * fraction),
            )
            raw = torch.maximum(raw, shifted * (1.0 - 0.32 * fraction))
        sigma = max(1.1, (0.008 + 0.005 * float(parameters[6])) * scale)
        return _gaussian_blur(raw, sigma)

    def _raw_bottom_envelope(
        self,
        guide: torch.Tensor,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        height, width = guide.shape[-2:]
        y_coordinates = torch.arange(
            height,
            device=guide.device,
            dtype=guide.dtype,
        ).view(1, 1, height, 1)
        active = guide.amax(dim=2, keepdim=True)
        bottom = torch.where(
            guide > 0,
            y_coordinates,
            y_coordinates.new_full((), -1.0),
        ).amax(dim=2, keepdim=True)
        bottom = bottom.clamp_min(0.0)

        curve_sigma = max((0.025 + 0.030 * float(parameters[0])) * width, 1.0)
        numerator = _gaussian_blur(
            bottom * active, sigma_y=0.0, sigma_x=curve_sigma
        )
        denominator = _gaussian_blur(
            active, sigma_y=0.0, sigma_x=curve_sigma
        )
        curve = numerator / denominator.clamp_min(self.epsilon)

        distance_below = y_coordinates - curve
        center_offset = (0.025 + 0.020 * float(parameters[1])) * height
        vertical_sigma = max(
            (0.032 + 0.020 * float(parameters[2])) * height,
            2.0,
        )
        vertical = torch.exp(
            -0.5 * ((distance_below - center_offset) / vertical_sigma).square()
        )
        vertical = vertical * (distance_below >= -0.004 * height)

        taper_sigma = max((0.025 + 0.025 * float(parameters[3])) * width, 1.0)
        horizontal = _gaussian_blur(
            active, sigma_y=0.0, sigma_x=taper_sigma
        )
        horizontal = horizontal / horizontal.amax().clamp_min(self.epsilon)
        raw = vertical * horizontal

        horizontal_sign = -1.0 if float(parameters[4]) < 0.5 else 1.0
        offset_dx = horizontal_sign * (0.040 + 0.040 * float(parameters[5])) * width
        offset_dy = (0.025 + 0.025 * float(parameters[6])) * height
        offset = _translate_integer(raw, round(offset_dx), round(offset_dy))
        return torch.maximum(raw, (0.35 + 0.20 * float(parameters[7])) * offset)

    def _finish_mask(
        self,
        raw: torch.Tensor,
        guide: torch.Tensor,
        foreground: torch.Tensor,
    ) -> torch.Tensor:
        """Exclude foreground, handle empty masks, and set target mask mass."""

        height, width = raw.shape[-2:]
        background = (~foreground).to(raw.dtype)
        candidate = raw.clamp_min(0.0) * background

        # Fall back to an adjacent ring, then to background if needed.
        scale = min(height, width)
        fallback_ring = (
            _elliptical_dilate(
                guide,
                max(round(0.08 * scale), 1),
                max(round(0.12 * scale), 1),
            )
            - guide
        ).clamp_min(0.0)
        fallback_ring = _gaussian_blur(
            fallback_ring,
            sigma_y=max(1.0, 0.010 * scale),
        ) * background
        primary_empty = candidate.amax() <= self.epsilon
        candidate = torch.where(primary_empty, fallback_ring, candidate)
        ring_empty = candidate.amax() <= self.epsilon
        candidate = torch.where(ring_empty, background, candidate)

        candidate = candidate / candidate.amax().clamp_min(self.epsilon)
        low = candidate.new_tensor(0.03)
        high = candidate.new_tensor(40.0)
        foreground_fraction = foreground.to(candidate.dtype).mean()
        target_mean = torch.minimum(
            candidate.new_tensor(self.target_mask_mean),
            foreground_fraction * self.maximum_shadow_to_foreground_ratio,
        )
        # Adjust mass through the exponent while preserving the mask peak.
        for _ in range(40):
            exponent = 0.5 * (low + high)
            current_mean = candidate.pow(exponent).mean()
            needs_less_mass = current_mean > target_mean
            low = torch.where(needs_less_mass, exponent, low)
            high = torch.where(needs_less_mass, high, exponent)
        mask = candidate.pow(0.5 * (low + high)) * background
        return mask

    def make_mask(self, family_id, target, parameters):
        foreground = target > 0.0
        guide = foreground.to(target.dtype)
        generators = (self._raw_bottom_envelope, self._raw_lower_wrap, self._raw_contact)
        raw = generators[int(family_id)](guide, parameters)
        return self._finish_mask(raw, guide, foreground)


class BackgroundDarkening:
    def __init__(self, seed=812033, probability=0.5, strength_min=0.35, strength_max=0.70):
        if not 0 <= probability <= 1 or not 0 <= strength_min <= strength_max <= 1:
            raise ValueError('Invalid darkening parameters')
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.probability = probability
        self.strength_min, self.strength_max = strength_min, strength_max
        self.generator = BackgroundMaskGenerator()

    @torch.no_grad()
    def __call__(self, images, target):
        images, target = self.generator._validate_inputs(images, target)
        if not torch.all((target == 0) | (target == 1)):
            raise ValueError('Binary aligned GT is required')
        counts = target.flatten(1).sum(1).cpu()
        valid = (counts > 0) & (counts < target[0].numel())
        selected = (torch.rand(len(images), generator=self.rng) < self.probability) & valid
        if self.probability > 0 and not selected.any() and valid.any():
            eligible = valid.nonzero().flatten()
            selected[eligible[int(torch.randint(len(eligible), (1,), generator=self.rng))]] = True
        ids = selected.nonzero().flatten().to(images.device)
        if not len(ids):
            return ids, None, None
        families = self.generator._sample_families(len(ids), self.rng)
        strengths = self.strength_min + (self.strength_max - self.strength_min) * torch.rand(len(ids), generator=self.rng)
        parameters = torch.rand(len(ids), 12, generator=self.rng)
        x, y = images[ids], target[ids]
        masks = torch.cat([self.generator.make_mask(families[i], y[i:i+1], parameters[i]) for i in range(len(ids))])
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb = (x * std + mean).clamp(0, 1)
        delta = strengths.to(x.device)[:, None, None, None] * masks * torch.minimum(rgb, 1 - rgb)
        changed = torch.where(masks > 0, x - delta / std, x)
        if torch.count_nonzero(masks * y):
            raise AssertionError('Foreground mask leakage')
        return ids, changed, masks
