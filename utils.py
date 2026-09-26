"""Optimizer, EMA, reproducibility, and checkpoint utilities."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from models import DIMOUSOD


def build_optimizer(model):
    backbone, task = [], []
    for name, p in model.named_parameters():
        (backbone if name.startswith('backbone.') else task).append(p)
    groups = [dict(params=backbone, lr=5e-5, initial_lr=5e-5, min_lr=2.5e-7, name='backbone'),
              dict(params=task, lr=2e-4, initial_lr=2e-4, min_lr=1e-6, name='task'),
              dict(params=[], lr=5e-5, initial_lr=5e-5, min_lr=2.5e-7, name='backbone_no_decay', weight_decay=0.)]
    return torch.optim.AdamW(groups, betas=(.9, .99), eps=1e-8, weight_decay=1e-4, foreach=False)


def set_learning_rate(optimizer, step, total, warmup):
    if step < warmup:
        factor = .10 + .90 * (step + 1) / max(warmup, 1)
    else:
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.), 1.)
        factor = .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = group['min_lr'] + (group['initial_lr'] - group['min_lr']) * factor


class ModelEMA:
    def __init__(self, model):
        self.module = copy.deepcopy(model).eval()
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = .999 * (1 - math.exp(-self.updates / 2000.))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            other = source[name].detach()
            if name.endswith(('fixed_mean', 'fixed_scale')) or not value.is_floating_point():
                value.copy_(other)
            else:
                value.mul_(decay).add_(other, alpha=1 - decay)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def rng_state(generators):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loaders={key: gen.get_state() for key, gen in generators.items()})


def restore_rng(state, generators):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    for key, value in state['loaders'].items():
        generators[key].set_state(value)


def load_model(path, device='cpu', trusted_legacy=False):
    """Strictly load portable releases or a trusted original EI_ONLY checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=not trusted_legacy)
    if 'model_state_dict' not in checkpoint:
        raise ValueError('Expected a checkpoint containing model_state_dict')
    state = checkpoint['model_state_dict']
    compensation = any(key.startswith('compensation.') for key in state)
    arm = checkpoint.get('arm')
    if arm not in (None, 'EI_ONLY', 'Base', 'A0'):
        raise ValueError('This release supports EI_ONLY and Base weights, not other experimental arms')
    model = DIMOUSOD(compensation=compensation)
    model.load_state_dict(state, strict=True)
    if compensation:
        stats = checkpoint.get('fixed_statistics')
        if not stats:
            raise ValueError('Missing fixed photometric statistics in checkpoint')
        for key, field in [('fixed_mean', 'mean'), ('fixed_scale', 'scale')]:
            expected = torch.as_tensor(stats[field], dtype=torch.float32).view(1, 3, 1, 1)
            if not torch.equal(getattr(model.compensation, key).cpu(), expected):
                raise ValueError('Fixed statistics differ from checkpoint buffers; use the verified checkpoint')
    return model.to(device).eval(), checkpoint


def portable_checkpoint(saved):
    """Remove training paths, dataset names, and optimizer/RNG state for distribution."""
    stats = saved.get('fixed_statistics')
    if stats:
        stats = {key: stats[key] for key in ('mean', 'scale', 'std') if key in stats}
    return dict(format='dimo-usod-v1', arm=saved.get('arm', 'EI_ONLY'),
                epoch=int(saved.get('epoch', 0)), selection_weights=saved.get('selection_weights', 'EMA'),
                model_state_dict=saved['model_state_dict'], fixed_statistics=stats,
                input_size=320, cpcs_alpha=2.2)
