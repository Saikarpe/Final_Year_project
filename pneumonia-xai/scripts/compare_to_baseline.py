"""Side-by-side of the archived frozen-VGG16 baseline and the new M-1b model.

Both were evaluated on the SAME 582 test images (verified: identical split
membership, only 16 path strings differ because the Mendeley copy has no
val/ folder). So differences here are model differences, not resampling.
"""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
new = json.load(open(os.path.join(REPO, 'models', 'metrics.json'), encoding='utf-8'))
old = json.load(open(os.path.join(REPO, 'models', 'baseline_vgg16_2026-09-07', 'metrics.json'),
                     encoding='utf-8'))


def arrow(n, o, higher_better=True, pct=False):
    d = n - o
    if abs(d) < 1e-9:
        return '  ='
    good = (d > 0) == higher_better
    sign = '+' if d > 0 else ''
    scale = 100 if pct else 1
    return f'{sign}{d*scale:.2f}{"pp" if pct else ""} {"BETTER" if good else "worse"}'


print('=' * 78)
print('CLASSIFICATION  (same 582 test images)')
print('=' * 78)
print(f'{"metric":<26}{"VGG16 (frozen)":>18}{"DenseNet121 (M-1b)":>22}   delta')
oc, nc = old['classification'], new['classification']
rows = [
    ('AUROC', oc['auroc']['auroc'], nc['auroc']['auroc'], True, False),
    ('Sensitivity', oc['sensitivity_specificity']['sensitivity'],
     nc['sensitivity_specificity']['sensitivity'], True, True),
    ('Specificity', oc['sensitivity_specificity']['specificity'],
     nc['sensitivity_specificity']['specificity'], True, True),
    ('Brier score (lower=better)', oc['calibration']['brier_score'],
     nc['calibration']['brier_score'], False, False),
]
for name, o, n, hb, pct in rows:
    print(f'{name:<26}{o:>18.4f}{n:>22.4f}   {arrow(n, o, hb, pct)}')

print()
oss, nss = old['sensitivity_specificity'] if 'sensitivity_specificity' in old else oc['sensitivity_specificity'], nc['sensitivity_specificity']
print(f'{"Confusion":<26}{"VGG16":>18}{"DenseNet121":>22}')
for k, label in (('fn', 'False negatives (MISSED)'), ('fp', 'False positives'),
                 ('tp', 'True positives'), ('tn', 'True negatives')):
    o, n = oss[k], nss[k]
    d = n - o
    note = ''
    if k == 'fn':
        note = f'   {abs(d)} {"fewer" if d < 0 else "more"} missed pneumonias'
    print(f'{label:<26}{o:>18}{n:>22}{note}')

print(f'\nAUROC 95% CI   VGG16 [{oc["auroc"]["ci_lower"]:.4f}, {oc["auroc"]["ci_upper"]:.4f}]'
      f'   M-1b [{nc["auroc"]["ci_lower"]:.4f}, {nc["auroc"]["ci_upper"]:.4f}]')
if nc['auroc']['ci_lower'] > oc['auroc']['auroc']:
    print('  -> M-1b CI lower bound exceeds the VGG16 point estimate.')
else:
    print('  -> CIs overlap: the AUROC gain is NOT statistically separated on n=582.')
print(f'Decision threshold   VGG16 {oc["decision_threshold"]:.4f}   M-1b {nc["decision_threshold"]:.4f}')

print()
print('=' * 78)
print('EXPLANATION QUALITY  (deletion lower=better, insertion higher=better)')
print('=' * 78)
print(f'{"method":<24}{"del AUC":>10}{"ins AUC":>10}{"ROAD":>9}{"runtime s":>11}{"n":>5}')
for name, row in new['explanation'].items():
    rt = (new.get('runtime', {}).get(name) or {}).get('mean_seconds')
    def f(v, w=10, p=3):
        return f'{v:>{w}.{p}f}' if isinstance(v, (int, float)) else f'{"—":>{w}}'
    print(f'{name:<24}{f(row.get("deletion_auc_mean"))}{f(row.get("insertion_auc_mean"))}'
          f'{f(row.get("road_mean_drop"), 9)}{f(rt, 11, 1)}{row.get("n_samples", "—"):>5}')

print()
print('=' * 78)
print('CONFORMAL COVERAGE')
print('=' * 78)
for a, cov in new['uncertainty']['empirical_coverage'].items():
    target = 1 - float(a)
    ok = 'ok' if cov >= target - 0.03 else 'UNDER'
    print(f'  alpha={a}  target {target:.0%}  empirical {cov:.1%}   {ok}')

print()
print('=' * 78)
print('SANITY CHECKS  (similarity should FALL as the model is randomized)')
print('=' * 78)
for method, data in new['sanity_checks']['methods'].items():
    stages = data['cascading_randomization']['stages']

    def fmt(v):
        # None (JSON null) means the randomized model emitted a constant
        # heatmap, so the rank correlation is undefined rather than zero.
        # Older metrics.json files wrote NaN for the same case.
        if v is None or (isinstance(v, float) and v != v):
            return '  undef'
        return f'{v:+.3f}'

    first = fmt(stages[1]['similarity_to_original']) if len(stages) > 1 else '  undef'
    last = fmt(stages[-1]['similarity_to_original'])
    lab = fmt(data['label_randomization']['similarity_to_original'])
    n_undef = sum(1 for s_ in stages if fmt(s_['similarity_to_original']) == '  undef')
    print(f'  {method:<22} first-stage {first}  fully-random {last}  label-rand {lab}'
          f'   ({n_undef}/{len(stages)} stages undefined)')

print()
print(f'model_hash {new.get("model_hash")}   generated {new.get("generated_at")}')
