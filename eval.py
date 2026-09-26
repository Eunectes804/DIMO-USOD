"""Evaluate saved uint8 saliency maps with the paper's py_sod_metrics protocol."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
import py_sod_metrics
from utils import write_json


def meters():
    return {'S': py_sod_metrics.Smeasure(), 'E': py_sod_metrics.Emeasure(),
            'wF': py_sod_metrics.WeightedFmeasure(), 'MAE': py_sod_metrics.MAE()}


def values(metric):
    return dict(Sm=float(metric['S'].get_results()['sm']),
                meanEm=float(metric['E'].get_results()['em']['curve'].mean()),
                maxEm=float(metric['E'].get_results()['em']['curve'].max()),
                wF=float(metric['wF'].get_results()['wfm']),
                MAE=float(metric['MAE'].get_results()['mae']))


def image_map(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            if path.stem in result:
                raise ValueError('Duplicate image stem: ' + path.stem)
            result[path.stem] = path
    if not result:
        raise ValueError('No images: ' + str(directory))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-dir', type=Path, required=True)
    p.add_argument('--gt-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/evaluation'))
    a = p.parse_args()
    preds, gts = image_map(a.pred_dir), image_map(a.gt_dir)
    if preds.keys() != gts.keys():
        raise ValueError('Prediction/GT mismatch. Missing predictions: {}; extra predictions: {}'.format(
            sorted(gts.keys() - preds.keys())[:10], sorted(preds.keys() - gts.keys())[:10]))
    total, rows = meters(), []
    for i, name in enumerate(sorted(gts)):
        with Image.open(preds[name]) as source:
            pred = np.asarray(source.convert('L'), dtype=np.uint8)
        with Image.open(gts[name]) as source:
            gt = np.asarray(source.convert('L'), dtype=np.uint8)
        if pred.shape != gt.shape:
            raise ValueError('Prediction and GT resolution differ: ' + name)
        gt = np.ascontiguousarray((gt >= 128).astype(np.uint8) * 255)
        pred = np.ascontiguousarray(pred)
        one = meters()
        for metric in (total, one):
            for instance in metric.values():
                instance.step(pred=pred, gt=gt)
        rows.append(dict(name=name, **values(one)))
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(gts):
            print('Evaluation {}/{}'.format(i + 1, len(gts)), flush=True)
    report = dict(images=len(rows), **values(total), pysodmetrics=importlib.metadata.version('pysodmetrics'),
                  protocol='uint8 predictions at GT resolution; binary GT >=128; py_sod_metrics defaults including per-image prediction normalization',
                  native_grid_metrics=False)
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output / 'metrics.json', report)
    with (a.output / 'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)


if __name__ == '__main__':
    main()
