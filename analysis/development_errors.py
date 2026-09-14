"""Describe a fixed selected system on DEV; no fitting or model selection.

Manifest: cxi/ixi/cxc objects with `labels`, `probabilities` paths;
ixi additionally has `reversed_probabilities`. Inputs are local saved DEV files.
Only aggregate metrics and input hashes are written, not row IDs or labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

CLASSES = ['same_figure', 'same_paper', 'related_papers', 'unrelated_papers']
COLS = ['p_' + c for c in CLASSES]


def metrics(cm):
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(4), where=(support + predicted) != 0)
    return dict(confusion=cm.tolist(), support=support.tolist(), f1=f1.tolist(),
                macro_f1=float(f1.mean()), active_class_macro_f1=float(f1[support > 0].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    hashes, results, seen = {}, {}, set()
    pooled = np.zeros((4, 4), dtype=np.int64)
    for task, count in [('cxi', 794), ('ixi', 590), ('cxc', 602)]:
        spec = manifest[task]
        frames = {}
        for role, path in spec.items():
            p = Path(path)
            hashes[task + '/' + role] = hashlib.sha256(p.read_bytes()).hexdigest()
            frame = pl.read_parquet(p)
            assert frame.height == count and frame['pair_id'].n_unique() == count
            frames[role] = frame.sort('pair_id')
        labels = frames['labels']
        ids = labels['pair_id'].to_list()
        assert not seen.intersection(ids)
        seen.update(ids)
        arrays = []
        for role in ['probabilities'] + (['reversed_probabilities'] if task == 'ixi' else []):
            frame = frames[role]
            assert frame['pair_id'].to_list() == ids
            prob = frame.select(COLS).to_numpy()
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            assert np.allclose(prob.sum(axis=1), 1, atol=1e-6)
            if task != 'cxi':
                assert np.all(prob[:, 0] == 0)
            arrays.append(prob)
        y = labels['y'].to_numpy()
        assert np.isin(y, range(4)).all()
        pred = np.mean(arrays, axis=0).argmax(axis=1)
        cm = np.zeros((4, 4), dtype=np.int64)
        np.add.at(cm, (y, pred), 1)
        pooled += cm
        results[task] = metrics(cm)
    results['pooled'] = metrics(pooled)
    assert abs(results['pooled']['macro_f1'] - 0.7562033665434074) < 1e-12
    output = dict(classes=CLASSES, confusion_axes='rows=true, columns=predicted',
                  scope='Post-hoc description of repeatedly selected DEV results; not independent evaluation',
                  input_sha256=hashes, results=results)
    args.out.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
