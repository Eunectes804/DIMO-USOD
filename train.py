"""Train DIMO-USOD or the clean Base with the final 50-epoch protocol."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import DIMOUSOD
from augmentation import BackgroundDarkening
from dataset import USOD10KDataset, make_loader
from losses import SegmentationLoss, BCEWithDiceLoss
from models.dimo import photometric_descriptor
from utils import (ModelEMA, build_optimizer, set_learning_rate,
                   seed_all, write_json, save_checkpoint, rng_state, restore_rng, sha256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('dimo', 'base'), default='dimo')
    parser.add_argument('--data-root', type=Path, default=Path('data/USOD10K'))
    parser.add_argument('--backbone-pretrained', type=Path, default=Path('pretrained/tinynext_s.pth'))
    parser.add_argument('--fixed-statistics', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/dimo_seed2026'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train-sizes', type=int, nargs='+', default=[256, 320, 352])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--aux-start-epoch', type=int, default=3)
    parser.add_argument('--aux-ramp-epochs', type=int, default=5)
    parser.add_argument('--view-probability', type=float, default=0.5)
    parser.add_argument('--view-weight', type=float, default=0.1)
    parser.add_argument('--no-darkening', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume output/last.pth from this trusted run')
    parser.add_argument('--stop-after-epoch', type=int, default=0, help='Stop early without changing the full schedule')
    parser.add_argument('--train-limit', type=int, default=0, help='Smoke test only')
    parser.add_argument('--eval-limit', type=int, default=0, help='Smoke test only')
    parser.add_argument('--max-train-batches', type=int, default=0, help='Smoke test only, per scale')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--random-init', action='store_true', help='Smoke tests only; not the paper protocol')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.accumulation_steps,
           args.aux_start_epoch, args.aux_ramp_epochs, args.log_every) < 1 or args.batch_size < 2:
        parser.error('Positive counts and physical batch size >= 2 required')
    if min(args.workers, args.warmup_epochs, args.train_limit, args.eval_limit, args.max_train_batches,
           args.stop_after_epoch) < 0:
        parser.error('Negative count')
    if not args.train_sizes or any(s < 64 or s % 32 for s in args.train_sizes) or len(set(args.train_sizes)) != len(args.train_sizes):
        parser.error('Distinct training sizes >=64 and divisible by 32 required')
    if not 0 <= args.view_probability <= 1 or not math.isfinite(args.view_weight) or args.view_weight < 0:
        parser.error('Invalid augmentation probability/weight')
    return args


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mae, iou, count = 0., 0., 0
    for batch in loader:
        p = model(batch['image'].to(device))['final_logits'].float().sigmoid()
        y = batch['mask']
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite validation prediction')
        for probability, target in zip(p.cpu().numpy(), y.numpy()):
            pred, truth = probability >= .5, target.astype(bool)
            intersection, union = int((pred & truth).sum()), int((pred | truth).sum())
            mae += float(np.abs(probability.astype(np.float64) - truth).mean())
            iou += (intersection + 1e-6) / (union + 1e-6)
            count += 1
    return dict(mae=mae / count, iou=iou / count, sample_count=count,
                selection_score=(iou / count + 5 * (1 - mae / count)) / 6)


def prepare_fixed_statistics(data_root, output, workers):
    """Compute frozen photometric statistics from clean training RGB only."""
    dataset = USOD10KDataset(data_root, 'train', 320, augment=False)
    loader = DataLoader(dataset, batch_size=16, num_workers=workers, shuffle=False)
    sums = np.zeros(3, dtype=np.float64)
    squares = sums.copy()
    pixels = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            descriptor = photometric_descriptor(batch['image'].float(), (80, 80)).double()
            sums += descriptor.sum((0, 2, 3)).numpy()
            squares += descriptor.square().sum((0, 2, 3)).numpy()
            pixels += descriptor.shape[0] * 80 * 80
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(loader):
                print('Statistics: {}/{} batches'.format(index + 1, len(loader)), flush=True)
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 0))
    record = dict(mean=mean.tolist(), std=std.tolist(), scale=(std + 1e-6).tolist(),
                  contract=dict(split='train', images=len(dataset), input_size=320,
                                descriptor_size=[80, 80], augment=False,
                                names=[p.stem for p, _ in dataset.samples]))
    write_json(output, record)
    return record


def main():
    a = parse_args()
    torch.set_num_threads(4)
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select --device cpu')
    seed_all(a.seed)
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise FileExistsError('Output is not empty; choose a new directory or use --resume')
    a.output.mkdir(parents=True, exist_ok=True)
    model = DIMOUSOD(None if a.random_init else str(a.backbone_pretrained),
                    compensation=a.model == 'dimo', module_seed=a.seed + 620003).to(a.device)
    statistics = None
    if a.model == 'dimo':
        path = a.fixed_statistics or a.output / 'fixed_statistics.json'
        statistics = json.loads(path.read_text(encoding='utf-8')) if path.exists() else prepare_fixed_statistics(a.data_root, path, a.workers)
        if statistics.get('contract', {}).get('split') != 'train':
            raise ValueError('Fixed statistics must be computed from clean training RGB')
        model.compensation.set_fixed_statistics(statistics)
    optimizer, ema = build_optimizer(model), ModelEMA(model)
    criterion, auxiliary_loss = SegmentationLoss(), BCEWithDiceLoss()
    seed_all(a.seed)
    loaders, generators = {}, {}
    for size in a.train_sizes:
        loaders[size], generators[str(size)], names = make_loader(a.data_root, 'train', size, a.batch_size, a.workers, a.seed, a.train_limit)
    validation, generators['val'], _ = make_loader(a.data_root, 'val', 320, a.eval_batch_size, a.workers, a.seed, a.eval_limit)
    if statistics and not a.train_limit and statistics['contract'].get('names') != names:
        raise ValueError('Statistics and training sample names differ')
    view = BackgroundDarkening(a.seed + 810007, a.view_probability)
    batches = sum(min(len(loader), a.max_train_batches) if a.max_train_batches else len(loader) for loader in loaders.values())
    updates = math.ceil(batches / a.accumulation_steps)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    source_root = Path(__file__).resolve().parent
    source_files = list((source_root / 'models').glob('*.py')) + [
        source_root / name for name in ('train.py', 'dataset.py', 'augmentation.py', 'losses.py', 'utils.py')]
    code_digest = {p.relative_to(source_root).as_posix(): sha256(p) for p in sorted(source_files)}
    history, first, global_step, best, best_epoch = [], 1, 0, -1., 0
    if a.resume:
        saved = torch.load(a.output / 'last.pth', map_location='cpu', weights_only=False)
        for key, value in arguments.items():
            if key not in ('resume', 'stop_after_epoch', 'log_every', 'device') and saved['arguments'][key] != value:
                raise ValueError('Resume argument differs: ' + key)
        if saved['source_sha256'] != code_digest:
            raise ValueError('Source changed since checkpoint; do not mix implementations when resuming')
        model.load_state_dict(saved['student_state_dict'], strict=True)
        ema.module.load_state_dict(saved['model_state_dict'], strict=True)
        ema.updates = saved['ema_updates']
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, first, global_step = saved['history'], saved['epoch'] + 1, saved['global_step']
        best, best_epoch = saved['best_score'], saved['best_epoch']
        restore_rng(saved['rng_state'], generators)
        view.rng.set_state(saved['view_rng'])
        print('Resumed epoch {}, optimizer step {}'.format(first - 1, global_step), flush=True)
    else:
        write_json(a.output / 'protocol.json', dict(arguments=arguments, source_sha256=code_digest,
            parameters=sum(p.numel() for p in model.parameters()), primary='final epoch EMA',
            secondary='best raw validation score', precision='FP32, no AMP/TF32',
            smoke=bool(a.random_init or a.train_limit or a.eval_limit or a.max_train_batches)))
    print('Model={}, parameters={:,}; final EMA is primary; best_score EMA is secondary'.format(a.model, sum(p.numel() for p in model.parameters())), flush=True)

    def payload(epoch):
        return dict(format='dimo-usod-v1', arm='EI_ONLY' if a.model == 'dimo' else 'Base',
                    epoch=epoch, selection_weights='EMA', model_state_dict=ema.module.state_dict(),
                    fixed_statistics=statistics, arguments=arguments, source_sha256=code_digest,
                    input_size=320, cpcs_alpha=2.2)

    stop = min(a.epochs, a.stop_after_epoch) if a.stop_after_epoch else a.epochs
    for epoch in range(first, stop + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aux_weight = 0. if a.no_darkening else a.view_weight * min(1., max(0., (epoch - a.aux_start_epoch + 1) / a.aux_ramp_epochs))
        batch_index, accumulation, loss_sum, sample_count, views = 0, 0, 0., 0, 0
        for size, loader in loaders.items():
            for index, batch in enumerate(loader):
                if a.max_train_batches and index >= a.max_train_batches:
                    break
                if accumulation == 0:
                    set_learning_rate(optimizer, global_step, a.epochs * updates, a.warmup_epochs * updates)
                x = batch['image'].to(a.device, non_blocking=True)
                targets = batch['granularity_mask'].to(a.device, non_blocking=True)
                ids, altered = [], None
                if aux_weight > 0:
                    ids, altered, _ = view(x, targets[:, 0:1])
                outputs = model(torch.cat((x, altered)) if len(ids) else x)
                clean_outputs = {key: value[:len(x)] for key, value in outputs.items()}
                clean = criterion(clean_outputs, targets)
                aux = auxiliary_loss(outputs['final_logits'][len(x):].float(), targets[ids, 0:1].float()) if len(ids) else clean * 0.
                loss = clean + aux_weight * aux
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss / a.accumulation_steps).backward()
                batch_index += 1
                accumulation += 1
                if accumulation == a.accumulation_steps or batch_index == batches:
                    if accumulation != a.accumulation_steps:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.mul_(a.accumulation_steps / accumulation)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accumulation = 0
                loss_sum += float(loss.detach()) * len(x)
                sample_count += len(x)
                views += len(ids)
                if batch_index == 1 or batch_index % a.log_every == 0 or batch_index == batches:
                    print('Epoch {}/{} batch {}/{} size={} loss={:.5f} clean={:.5f} aux={:.5f} weight={:.3f} lr={:.2e}/{:.2e}'.format(
                        epoch, a.epochs, batch_index, batches, size, loss_sum / sample_count, float(clean), float(aux), aux_weight,
                        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']), flush=True)
        val = validate(ema.module, validation, a.device)
        score = val['selection_score']
        row = dict(epoch=epoch, train_loss=loss_sum / sample_count, selected_views=views,
                   aux_weight=aux_weight, validation=val, elapsed_seconds=time.perf_counter() - start)
        history.append(row)
        if score > best:
            best, best_epoch = score, epoch
            save_checkpoint(a.output / 'best_score.pth', payload(epoch))
        saved = payload(epoch)
        saved.update(student_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     ema_updates=ema.updates, history=history, global_step=global_step, best_score=best,
                     best_epoch=best_epoch, rng_state=rng_state(generators), view_rng=view.rng.get_state())
        save_checkpoint(a.output / 'last.pth', saved)
        write_json(a.output / 'history.json', history)
        print('EPOCH={} RAW MAE={:.6f} IoU={:.6f} Score={:.8f} BEST epoch={} score={:.8f} time={:.1f}s'.format(
            epoch, val['mae'], val['iou'], score, best_epoch, best, row['elapsed_seconds']), flush=True)
    if history and history[-1]['epoch'] == a.epochs:
        save_checkpoint(a.output / 'final.pth', payload(a.epochs))
        print('Finished. Primary: final.pth | Secondary: best_score.pth | Resume: last.pth', flush=True)


if __name__ == '__main__':
    main()
