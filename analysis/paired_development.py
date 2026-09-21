"""Fixed-model, paired DEV diagnostics; no training, tuning, or test data.

Input manifest has labels, groups, and models mappings. Each model supplies
path and optional expected_macro_f1. Probability files use canonical p_ columns;
generative files use predicted_label. Invalid generations abort, never drop rows.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from development_errors import CLASSES, COLS, metrics


def confusion(y, p):
    return np.bincount(4 * y + p, minlength=16).reshape(4, 4)


def batched_f1(cms):
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    den = cms.sum(axis=-1) + cms.sum(axis=-2)
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0).mean(axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=10000)
    args = ap.parse_args()
    m = json.loads(args.manifest.read_text())
    hashes = {}

    def read(key, path):
        p = Path(path)
        hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest()
        return pl.read_parquet(p)

    gold = read('labels', m['labels']).select('pair_id', 'y').sort('pair_id')
    assert gold.height == 794 and gold['pair_id'].n_unique() == 794
    groups = read('groups', m['groups']).select('pair_id', 'group_id', 'fold')
    assert groups['pair_id'].n_unique() == groups.height
    gold = gold.join(groups, on='pair_id', how='left', validate='1:1').sort('pair_id')
    assert gold['group_id'].null_count() == 0 and gold['fold'].to_list() == [1] * 794
    ids = gold['pair_id'].to_list()
    y = gold['y'].to_numpy()
    assert np.isin(y, range(4)).all()
    _, gi = np.unique(gold['group_id'].to_numpy(), return_inverse=True)
    ng = int(gi.max() + 1)
    preds, gcms, result = {}, {}, {}
    for name, spec in m['models'].items():
        x = read(name, spec['path']).sort('pair_id')
        assert x.height == 794 and x['pair_id'].to_list() == ids
        if all(c in x.columns for c in COLS):
            probs = x.select(COLS).to_numpy()
            assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
            assert np.allclose(probs.sum(1), 1, atol=1e-6)
            p = probs.argmax(1)
        else:
            receipt_path = Path(spec['metadata'])
            hashes[name + '/metadata'] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            receipt = json.loads(receipt_path.read_text())
            assert receipt['fallback'] == 0 and receipt['valid_canonical'] == 794 and receipt['eos_reached'] == 794
            labels = x['predicted_label'].to_list()
            assert all(v in CLASSES for v in labels), 'Invalid generation: inspect historical fallback protocol'
            p = np.array([CLASSES.index(v) for v in labels])
        preds[name] = p
        cm = confusion(y, p)
        result[name] = metrics(cm)
        result[name]['prediction_counts'] = cm.sum(axis=0).tolist()
        if 'metadata' in spec:
            result[name]['generation_integrity'] = dict(fallback=0, valid_canonical=794, eos_reached=794)
        if 'expected_macro_f1' in spec:
            assert abs(result[name]['macro_f1'] - spec['expected_macro_f1']) < 1e-9
        counts = np.zeros((ng, 4, 4), dtype=int)
        np.add.at(counts, (gi, y, p), 1)
        gcms[name] = counts
        mask = np.isin(y, [2, 3])
        result[name]['related_unrelated_true_subset'] = {
            'n': int(mask.sum()), 'correct_original_four_way': int((p[mask] == y[mask]).sum()),
            'predicted_outside_subset': int((p[mask] < 2).sum()),
            'note': 'Original four-way decisions retained; no binary renormalization'}
    fixed_path = Path(m['fixed_other_modalities'])
    hashes['fixed_other_modalities'] = hashlib.sha256(fixed_path.read_bytes()).hexdigest()
    fixed = json.loads(fixed_path.read_text())['results']
    other = np.array(fixed['ixi']['confusion']) + np.array(fixed['cxc']['confusion'])
    for name, p in preds.items():
        whole = metrics(confusion(y, p) + other)['macro_f1']
        result[name]['whole_macro_f1_fixed_ixi_cxc'] = whole
        if 'expected_whole' in m['models'][name]:
            assert abs(whole - m['models'][name]['expected_whole']) < 1e-9
    # Paired inherited-component resampling; identical cluster draws for all models.
    rng = np.random.default_rng(20260921)
    samples = {name: [] for name in preds}
    for offset in range(0, args.draws, 250):
        draws = min(250, args.draws - offset)
        weights = rng.multinomial(ng, np.full(ng, 1 / ng), size=draws)
        for name, counts in gcms.items():
            cms = (weights @ counts.reshape(ng, 16)).reshape(draws, 4, 4)
            samples[name].append(batched_f1(cms))
    samples = {name: np.concatenate(v) for name, v in samples.items()}
    comparisons = {}
    for a, b in m['comparisons']:
        pa, pb = preds[a], preds[b]
        changed = pa != pb
        fixed = (pa != y) & (pb == y)
        broken = (pa == y) & (pb != y)
        both_wrong = changed & (pa != y) & (pb != y)
        diffs = samples[b] - samples[a]
        comparisons[a + ' -> ' + b] = dict(
            changed=int(changed.sum()), corrected=int(fixed.sum()), broken=int(broken.sum()),
            changed_both_wrong=int(both_wrong.sum()), net_correct=int(fixed.sum()-broken.sum()),
            corrected_by_true_class=np.bincount(y[fixed], minlength=4).tolist(),
            broken_by_true_class=np.bincount(y[broken], minlength=4).tolist(),
            delta_macro_f1=result[b]['macro_f1']-result[a]['macro_f1'],
            paired_cluster_percentile_interval_95=np.quantile(diffs, [.025, .975]).tolist(),
            delta_class_f1=(np.array(result[b]['f1'])-np.array(result[a]['f1'])).tolist())
    out = dict(scope='Post-hoc fixed-model comparisons on repeatedly selected CXI DEV; no independent confirmation',
               classes=CLASSES, n=794, inherited_groups=ng, bootstrap_draws=args.draws, bootstrap_seed=20260921,
               uncertainty='Conditional sampling variation only; not selection-corrected, training-seed uncertainty, or proof of causal superiority. Inherited groups may miss dependencies.',
               input_sha256=hashes, models=result, comparisons=comparisons)
    args.out.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
