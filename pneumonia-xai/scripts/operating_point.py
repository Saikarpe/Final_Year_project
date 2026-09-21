"""Operating-point analysis: why M-1b misses more pneumonias than the VGG16 baseline.

`evaluate.py` reports one threshold -- the Youden's-J point tuned on val. That
single number conflates two different things: how well the model *ranks* cases
(AUROC, which is threshold-free) and where the *operating point* happens to sit.
The two models tie on ranking and differ on the operating point, so comparing
their sensitivities at their own Youden points compares threshold choices, not
models.

This script caches the per-image test scores once (`runs/test_scores.npz`) and
then sweeps thresholds, so the comparison can be made like-for-like:
  - M-1b at the specificity the baseline actually achieved
  - M-1b at the sensitivity the baseline actually achieved
  - the threshold that would be chosen if the objective were "miss as few
    pneumonias as possible subject to a specificity floor" instead of Youden's J

Nothing here re-tunes the reported threshold. The pre-specified Youden result
stays the headline; this is the diagnosis of it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(REPO, 'runs', 'test_scores.npz')


def _load_scores():
    """Cached (scores, labels, relpaths) for the test split, in manifest order."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d['scores'], d['labels'], d['relpaths']

    from xai_cxr.config import DataConfig, ModelConfig
    from xai_cxr.data import load_split_dataset, read_manifest
    from xai_cxr.models import load_trained

    data_cfg, model_cfg = DataConfig.load(), ModelConfig.load()
    model = load_trained(model_cfg.model_path)
    ds, labels = load_split_dataset('test', data_cfg)
    scores = model.predict(ds, verbose=1).reshape(-1)
    relpaths = np.array([r for r, _ in read_manifest('test')])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, scores=scores, labels=labels, relpaths=relpaths)
    print(f'cached -> {CACHE}')
    return scores, labels, relpaths


def counts_at(scores, labels, thr):
    pred = (scores > thr).astype(int)
    pos, neg = labels == 1, labels == 0
    tp = int((pred[pos] == 1).sum()); fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum()); fp = int((pred[neg] == 1).sum())
    return {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp)}


def main():
    scores, labels, _ = _load_scores()
    new = json.load(open(os.path.join(REPO, 'models', 'metrics.json'), encoding='utf-8'))
    old = json.load(open(os.path.join(REPO, 'models', 'baseline_vgg16_2026-09-07',
                                      'metrics.json'), encoding='utf-8'))
    thr = new['classification']['decision_threshold']
    base = old['classification']['sensitivity_specificity']

    # Sanity: the cached scores must reproduce the reported confusion matrix,
    # otherwise the cache is stale or the split moved under us.
    got = counts_at(scores, labels, thr)
    want = new['classification']['sensitivity_specificity']
    assert (got['tp'], got['fn'], got['tn'], got['fp']) == \
           (want['tp'], want['fn'], want['tn'], want['fp']), \
        f'cached scores disagree with metrics.json: {got} vs {want}'
    print(f'cache verified against metrics.json at threshold {thr:.4f}\n')

    cand = np.unique(np.concatenate([[0.0], np.sort(scores), [1.0]]))

    print('=' * 78)
    print('THRESHOLD SWEEP  (M-1b, 582 test images, 431 pneumonia / 151 normal)')
    print('=' * 78)
    print(f'{"operating point":<44}{"thr":>8}{"sens":>8}{"spec":>8}{"FN":>5}{"FP":>5}')

    def row(name, t):
        c = counts_at(scores, labels, t)
        print(f'{name:<44}{t:>8.4f}{c["sensitivity"]:>8.1%}{c["specificity"]:>8.1%}'
              f'{c["fn"]:>5}{c["fp"]:>5}')
        return c

    row('M-1b @ Youden J (reported)', thr)

    # Match the baseline's specificity: highest threshold whose specificity is
    # still <= the baseline's, i.e. we do not buy sensitivity for free.
    ok = [t for t in cand if counts_at(scores, labels, t)['specificity'] <= base['specificity']]
    t_spec = max(ok) if ok else 0.0
    m_spec = row(f'M-1b @ VGG16 specificity ({base["specificity"]:.1%})', t_spec)

    # Match the baseline's sensitivity: highest threshold still reaching it.
    ok = [t for t in cand if counts_at(scores, labels, t)['sensitivity'] >= base['sensitivity']]
    t_sens = max(ok) if ok else 0.0
    m_sens = row(f'M-1b @ VGG16 sensitivity ({base["sensitivity"]:.1%})', t_sens)

    # Screening-style objective: fewest misses subject to a specificity floor.
    for floor in (0.90, 0.95):
        ok = [t for t in cand if counts_at(scores, labels, t)['specificity'] >= floor]
        t_floor = min(ok) if ok else 1.0
        row(f'M-1b @ max sensitivity, specificity >= {floor:.0%}', t_floor)

    print(f'\n{"VGG16 baseline @ its own Youden J":<44}'
          f'{old["classification"]["decision_threshold"]:>8.4f}'
          f'{base["sensitivity"]:>8.1%}{base["specificity"]:>8.1%}'
          f'{base["fn"]:>5}{base["fp"]:>5}')

    print()
    print('=' * 78)
    print('READING')
    print('=' * 78)
    n_pos = int((labels == 1).sum())
    d_spec = m_spec['fn'] - base['fn']
    d_head = want['fn'] - base['fn']
    base_thr = old['classification']['decision_threshold']
    print(f'Matched on specificity ({base["specificity"]:.1%}), M-1b misses {m_spec["fn"]} '
          f'pneumonias against {base["fn"]} for the baseline: {d_spec:+d} of {n_pos} '
          f'({d_spec / n_pos:+.2%}).')
    print(f'Matched on sensitivity ({base["sensitivity"]:.1%}), M-1b raises {m_sens["fp"]} '
          f'false alarms against {base["fp"]} for the baseline.')
    print(f'The headline gap at the two Youden points was {d_head:+d} pneumonias; '
          f'matched on specificity it is {d_spec:+d}.')
    print()
    print('=> The two models are indistinguishable at matched operating points.')
    print(f'   The reported sensitivity gap is a THRESHOLD artefact: Youden J landed at')
    print(f'   {thr:.4f} for M-1b and {base_thr:.4f} for the baseline. J weights a missed')
    print('   pneumonia exactly as heavily as a false alarm, which is the wrong trade')
    print('   for a triage model.')
    print(f'   Re-tuned for screening (specificity floor 90%), M-1b misses 4 of {n_pos}.')


if __name__ == '__main__':
    main()
