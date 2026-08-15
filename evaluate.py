"""Drift-Sense evaluation harness.

    python evaluate.py --data data/train --out results/

Runs localize.py's pipeline over every pair in the manifest and reports the
accuracy CURVE (not a single number), accuracy broken down by how much
aperiodic information the reference actually contains -- the scientific
result: localisation accuracy is a function of available aperiodic
information, not of effort -- plus breakdowns by style and noise, per-stage
timing, tie-break rate, and a confidence/reject-option analysis.

Then calibrates lam and k by sweep and re-checks the winner on a FRESH set
generated with a different seed, which is the only way to know the constants
were not fitted to one draw.

Writes into --out: metrics.json, accuracy_curve.png,
accuracy_vs_aperiodic.png, confidence_vs_error.png, success_case.png,
failure_case.png, sweep.csv.  No output is written outside --out (the fresh
validation set is generated into --out/val_seedNNNN); CPython's own
__pycache__ is the only exception.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import localize as L

TOLERANCES = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
AP_BUCKETS = [(0.0, 0.05, 'pure lattice\n(<0.05)'),
              (0.05, 0.35, 'sparse\n(0.05-0.35)'),
              (0.35, 0.65, 'moderate\n(0.35-0.65)'),
              (0.65, 1.01, 'rich\n(0.65-1.0)')]
SWEEP_LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
SWEEP_K = [1.0, 1.5, 2.0, 3.0]
# The brief's k grid starts at 1.0; measurements in Phase 7 put the optimum an
# order of magnitude below that (sigma = MAD(S) tracks the lattice OSCILLATION,
# not peak-height noise), so the sweep is extended downward rather than
# reporting the best of a grid that excludes the answer.
SWEEP_K_EXTRA = [0.05, 0.1, 0.5]


def load_pairs(data_dir):
    """Manifest records for a dataset directory.

    Shared by evaluate.py, make_demo.py, train_confidence.py and the gates, so
    the "you have not generated the data yet" case is answered once, with the
    command that fixes it, rather than as a bare traceback in four places.
    """
    path = os.path.join(data_dir, 'manifest.json')
    if not os.path.isfile(path):
        raise SystemExit(
            f"no manifest at {path}\n"
            f"Generate the dataset first, e.g.:\n"
            f"  {sys.executable} generate_dataset.py --style both --n 40 "
            f"--out {data_dir} --seed 0")
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise SystemExit(f"{path} contains no pairs")
    return records


def read_pair(data_dir, rec):
    return (L.load_grey(os.path.join(data_dir, rec['ref_path'])),
            L.load_grey(os.path.join(data_dir, rec['search_path'])))


def evaluate_set(data_dir, records, lam=None, k=None, keep_maps=False,
                 progress=False):
    """Run the pipeline over every pair.  Returns a list of result dicts."""
    rows = []
    for i, rec in enumerate(records):
        ref_u8, search_u8 = read_pair(data_dir, rec)
        timings = {}
        t0 = time.time()
        x, y, diag = L.localize(ref_u8, search_u8, 0.1, lam=lam, k=k,
                                timings=timings,
                                return_score_map=keep_maps)
        wall = time.time() - t0
        err = float(np.hypot(x - rec['true_x'], y - rec['true_y']))
        res = diag.get('resolve', {})
        rsc = diag.get('rescore', {})
        rows.append({
            'id': rec['id'], 'style': rec['style'],
            'ref_path': rec['ref_path'], 'search_path': rec['search_path'],
            'aperiodic_level': rec['aperiodic_level'],
            'search_dose': rec['search_dose'], 'scale': rec['scale'],
            'true_x': rec['true_x'], 'true_y': rec['true_y'],
            'pred_x': x, 'pred_y': y, 'error_px': err, 'wall_s': wall,
            'timings': dict(timings), 'method': diag.get('method'),
            'tie_break_used': bool(res.get('tie_break_used', False)),
            'family_size': int(res.get('family_size', 0)),
            'resolve_confidence': float(res.get('confidence', float('nan'))),
            'rescore_margin': float(rsc.get('margin', float('nan'))),
            'ecc_cc': float(rsc.get('best_ecc_cc', float('nan'))),
            'transform': diag.get('transform', {}),
            'score_map': diag.get('score_map') if keep_maps else None,
            # full diagnostics for core.confidence feature extraction; stripped
            # before metrics.json is written (see main)
            'diag': diag,
        })
        if progress:
            print(f"    {rec['id']} ap={rec['aperiodic_level']:.2f} "
                  f"err={err:8.2f}px {wall*1000:5.0f}ms", flush=True)
    return rows


def acc_at(errs, tol):
    errs = np.asarray(errs, dtype=float)
    return float(np.mean(errs <= tol)) if errs.size else float('nan')


def curve(errs):
    return {str(t): acc_at(errs, t) for t in TOLERANCES}


def summarise(rows):
    errs = np.array([r['error_px'] for r in rows])
    walls = np.array([r['wall_s'] for r in rows])
    stages = sorted({s for r in rows for s in r['timings']})
    stage_ms = {s: float(np.mean([r['timings'].get(s, 0.0)
                                  for r in rows]) * 1000) for s in stages}

    out = {
        'n_pairs': len(rows),
        'accuracy_curve': curve(errs),
        'error_px': {'median': float(np.median(errs)),
                     'mean': float(errs.mean()),
                     'p90': float(np.percentile(errs, 90)),
                     'p95': float(np.percentile(errs, 95)),
                     'max': float(errs.max())},
        'wall_s': {'mean': float(walls.mean()),
                   'median': float(np.median(walls)),
                   'p95': float(np.percentile(walls, 95)),
                   'max': float(walls.max())},
        'stage_ms': stage_ms,
    }

    # subpixel quality where the correct unit cell was chosen
    good = errs[errs <= 15.0]
    out['on_correct_cell'] = {
        'n': int(good.size), 'frac': float(good.size / max(len(rows), 1)),
        'median_px': float(np.median(good)) if good.size else None,
        'p90_px': float(np.percentile(good, 90)) if good.size else None}

    # THE headline: accuracy as a function of available aperiodic information
    by_ap = []
    for lo, hi, label in AP_BUCKETS:
        sub = [r for r in rows if lo <= r['aperiodic_level'] < hi]
        if not sub:
            continue
        e = np.array([r['error_px'] for r in sub])
        by_ap.append({'bucket': label.replace('\n', ' '), 'lo': lo, 'hi': hi,
                      'n': len(sub), 'accuracy_curve': curve(e),
                      'median_px': float(np.median(e))})
    out['by_aperiodic_level'] = by_ap

    for key, field in (('by_style', 'style'),):
        groups = {}
        for value in sorted({r[field] for r in rows}):
            sub = [r for r in rows if r[field] == value]
            e = np.array([r['error_px'] for r in sub])
            groups[str(value)] = {'n': len(sub), 'accuracy_curve': curve(e),
                                  'median_px': float(np.median(e))}
        out[key] = groups

    # noise: split at the median search dose (lower dose == noisier)
    doses = np.array([r['search_dose'] for r in rows])
    cut = float(np.median(doses))
    out['by_noise'] = {}
    for label, mask in (('noisier (dose <= %.0f)' % cut, doses <= cut),
                        ('cleaner (dose > %.0f)' % cut, doses > cut)):
        e = np.array([r['error_px'] for r, m in zip(rows, mask) if m])
        if e.size:
            out['by_noise'][label] = {'n': int(e.size),
                                      'accuracy_curve': curve(e),
                                      'median_px': float(np.median(e))}

    # tie-break rate and accuracy conditional on it
    tb = np.array([r['tie_break_used'] for r in rows])
    out['tie_break'] = {'rate': float(tb.mean())}
    for label, mask in (('tie_break_used', tb), ('resolved_outright', ~tb)):
        e = errs[mask]
        if e.size:
            out['tie_break'][label] = {'n': int(e.size),
                                       'accuracy_curve': curve(e),
                                       'median_px': float(np.median(e))}

    # reject option: a fab tool would decline the least-confident cases
    out['reject_option'] = {}
    for name in ('rescore_margin', 'ecc_cc', 'resolve_confidence'):
        conf = np.array([r[name] for r in rows], dtype=float)
        ok = np.isfinite(conf)
        if ok.sum() < 4:
            continue
        c, e = conf[ok], errs[ok]
        order = np.argsort(-c)                      # most confident first
        entry = {'spearman_vs_error': float(_spearman(c, e))}
        for frac in (0.9, 0.75, 0.5):
            keep = order[:max(1, int(round(frac * order.size)))]
            entry[f'top_{int(frac*100)}pct'] = {
                'n': int(keep.size), 'accuracy_curve': curve(e[keep]),
                'median_px': float(np.median(e[keep]))}
        out['reject_option'][name] = entry
    return out


def _rankdata(x):
    """Ranks with ties averaged.

    Plain argsort-of-argsort breaks ties arbitrarily, which biases the
    correlation whenever a signal is discrete -- family_size and ecc_cc both
    tie routinely here -- so the reported rho would depend on input order.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1.0, x.size + 1.0)
    s = x[order]
    i = 0
    while i < s.size:                      # average each run of equal values
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float('nan')


# ----------------------------------------------------------------- plots ---
def plot_accuracy_curve(rows, metrics, path):
    errs = np.array([r['error_px'] for r in rows])
    grid = np.logspace(np.log10(0.05), np.log10(500), 300)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(grid, [acc_at(errs, t) * 100 for t in grid], lw=2.5,
            color='#1f4e79', label=f'all pairs (n={len(rows)})')
    for style, colour in (('dram', '#d1495b'), ('finfet', '#2a9d8f')):
        e = np.array([r['error_px'] for r in rows if r['style'] == style])
        if e.size:
            ax.plot(grid, [acc_at(e, t) * 100 for t in grid], lw=1.5, ls='--',
                    color=colour, label=f'{style} (n={e.size})')
    for t in TOLERANCES:
        a = acc_at(errs, t) * 100
        ax.plot([t], [a], 'o', color='#1f4e79', ms=6)
        ax.annotate(f'{a:.0f}%', (t, a), textcoords='offset points',
                    xytext=(4, -12), fontsize=9, color='#1f4e79')
    ax.set_xscale('log')
    ax.set_xlabel('tolerance (px, log scale)')
    ax.set_ylabel('pairs localised within tolerance (%)')
    ax.set_title('Localisation accuracy vs tolerance')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_accuracy_vs_aperiodic(rows, path):
    labels, data = [], {t: [] for t in (2.0, 5.0, 25.0)}
    counts = []
    for lo, hi, label in AP_BUCKETS:
        sub = [r['error_px'] for r in rows
               if lo <= r['aperiodic_level'] < hi]
        if not sub:
            continue
        labels.append(label)
        counts.append(len(sub))
        for t in data:
            data[t].append(acc_at(sub, t) * 100)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                  gridspec_kw={'width_ratios': [1.15, 1]})
    xs = np.arange(len(labels))
    w = 0.26
    for i, (t, colour) in enumerate(zip((2.0, 5.0, 25.0),
                                        ('#1f4e79', '#2a9d8f', '#e9c46a'))):
        ax.bar(xs + (i - 1) * w, data[t], w, label=f'within {t:g} px',
               color=colour)
    for i, (x, n) in enumerate(zip(xs, counts)):
        ax.annotate(f'n={n}', (x, 2), ha='center', fontsize=9, color='#444')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('pairs localised (%)')
    ax.set_ylim(0, 100)
    ax.set_xlabel('aperiodic information in the layout')
    ax.set_title('Accuracy vs available aperiodic information')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ap = np.array([r['aperiodic_level'] for r in rows])
    er = np.array([r['error_px'] for r in rows])
    ax2.scatter(ap, np.maximum(er, 0.02), s=32, alpha=0.75,
                c=['#d1495b' if r['style'] == 'dram' else '#2a9d8f'
                   for r in rows])
    ax2.axhline(5.0, ls='--', color='#666', lw=1)
    ax2.annotate('5 px tolerance', (0.02, 5.6), fontsize=9, color='#666')
    ax2.set_yscale('log')
    ax2.set_xlabel('aperiodic_level')
    ax2.set_ylabel('localisation error (px, log)')
    ax2.set_title('Per-pair error (red DRAM, teal FinFET)')
    ax2.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    conf = np.array([r['rescore_margin'] for r in rows], dtype=float)
    errs = np.array([r['error_px'] for r in rows])
    ok = np.isfinite(conf)
    axes[0].scatter(conf[ok], np.maximum(errs[ok], 0.02), s=32, alpha=0.75,
                    color='#1f4e79')
    axes[0].axhline(5.0, ls='--', color='#666', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('confidence (rescore margin)')
    axes[0].set_ylabel('error (px, log)')
    axes[0].set_title('Confidence vs error')
    axes[0].grid(alpha=0.3, which='both')

    order = np.argsort(-conf[ok])
    e = errs[ok][order]
    fracs = np.linspace(0.1, 1.0, 40)
    for tol, colour in ((5.0, '#2a9d8f'), (25.0, '#e9c46a')):
        ys = [acc_at(e[:max(1, int(round(f * e.size)))], tol) * 100
              for f in fracs]
        axes[1].plot(fracs * 100, ys, lw=2, color=colour,
                     label=f'within {tol:g} px')
    axes[1].set_xlabel('% of predictions kept (most confident first)')
    axes[1].set_ylabel('accuracy on kept subset (%)')
    axes[1].set_title('Reject option: accuracy vs coverage')
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_case(data_dir, rec_row, path, title):
    ref_u8, search_u8 = read_pair(data_dir, rec_row)
    S = rec_row.get('score_map')
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    axes[0].imshow(ref_u8, cmap='gray')
    axes[0].set_title(f"reference ({rec_row['style']}, "
                      f"ap={rec_row['aperiodic_level']:.2f})")
    axes[1].imshow(search_u8, cmap='gray')
    axes[1].plot([rec_row['true_x']], [rec_row['true_y']], 'o', ms=14,
                 mfc='none', mec='#2ecc71', mew=2.5, label='true')
    axes[1].plot([rec_row['pred_x']], [rec_row['pred_y']], 'x', ms=14,
                 color='#e74c3c', mew=2.5, label='predicted')
    axes[1].legend(loc='upper right')
    axes[1].set_title(f"search — error {rec_row['error_px']:.2f} px")
    if S is not None:
        Sv = np.where(np.isfinite(S), S, np.nan)
        im = axes[2].imshow(Sv, cmap='viridis')
        axes[2].plot([rec_row['true_x']], [rec_row['true_y']], 'o', ms=14,
                     mfc='none', mec='#2ecc71', mew=2.5)
        axes[2].plot([rec_row['pred_x']], [rec_row['pred_y']], 'x', ms=14,
                     color='#e74c3c', mew=2.5)
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        axes[2].set_title(f"score surface (family={rec_row['family_size']}, "
                          f"tie_break={rec_row['tie_break_used']})")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------- calibration ---
def sweep(data_dir, records, out_dir, lams, ks):
    """Sweep lam x k, scoring accuracy at 5 px tolerance."""
    table, best = [], None
    total = len(lams) * len(ks)
    for i, lam in enumerate(lams):
        for j, k in enumerate(ks):
            rows = evaluate_set(data_dir, records, lam=lam, k=k)
            errs = np.array([r['error_px'] for r in rows])
            entry = {'lam': lam, 'k': k,
                     'acc_5px': acc_at(errs, 5.0),
                     'acc_2px': acc_at(errs, 2.0),
                     'acc_25px': acc_at(errs, 25.0),
                     'median_px': float(np.median(errs)),
                     'tie_break_rate': float(np.mean(
                         [r['tie_break_used'] for r in rows]))}
            table.append(entry)
            if best is None or entry['acc_5px'] > best['acc_5px'] or (
                    entry['acc_5px'] == best['acc_5px'] and
                    entry['median_px'] < best['median_px']):
                best = entry
            print(f"    [{i*len(ks)+j+1:2d}/{total}] lam={lam:<5.2f} k={k:<5.2f} "
                  f"acc@5px={entry['acc_5px']*100:5.1f}%  "
                  f"median={entry['median_px']:8.2f}px  "
                  f"ties={entry['tie_break_rate']*100:3.0f}%", flush=True)
    with open(os.path.join(out_dir, 'sweep.csv'), 'w') as f:
        f.write('lam,k,acc_2px,acc_5px,acc_25px,median_px,tie_break_rate\n')
        for e in table:
            f.write(f"{e['lam']},{e['k']},{e['acc_2px']:.4f},"
                    f"{e['acc_5px']:.4f},{e['acc_25px']:.4f},"
                    f"{e['median_px']:.4f},{e['tie_break_rate']:.4f}\n")
    return table, best


def print_report(m, title):
    print(f"\n{title}")
    print(f"  pairs: {m['n_pairs']}")
    print("  accuracy curve:      " + "  ".join(
        f"<={t:g}px {m['accuracy_curve'][str(t)]*100:5.1f}%" for t in TOLERANCES))
    e = m['error_px']
    print(f"  error px: median={e['median']:.2f} p90={e['p90']:.2f} "
          f"p95={e['p95']:.2f} max={e['max']:.1f}")
    c = m['on_correct_cell']
    if c['median_px'] is not None:
        print(f"  correct cell ({c['n']}/{m['n_pairs']}): "
              f"median {c['median_px']:.3f} px, p90 {c['p90_px']:.3f} px")
    w = m['wall_s']
    print(f"  wall/pair: mean={w['mean']*1000:.0f}ms median="
          f"{w['median']*1000:.0f}ms p95={w['p95']*1000:.0f}ms")
    print("  stages (mean ms): " + "  ".join(
        f"{k}={v:.0f}" for k, v in sorted(m['stage_ms'].items(),
                                          key=lambda kv: -kv[1])))
    print("\n  ACCURACY BY APERIODIC INFORMATION (the headline):")
    for b in m['by_aperiodic_level']:
        print(f"    {b['bucket']:22s} n={b['n']:2d}  "
              f"<=2px {b['accuracy_curve']['2.0']*100:5.1f}%  "
              f"<=5px {b['accuracy_curve']['5.0']*100:5.1f}%  "
              f"<=25px {b['accuracy_curve']['25.0']*100:5.1f}%  "
              f"median {b['median_px']:.2f}px")
    print("  by style:")
    for s, v in m['by_style'].items():
        print(f"    {s:22s} n={v['n']:2d}  "
              f"<=5px {v['accuracy_curve']['5.0']*100:5.1f}%  "
              f"median {v['median_px']:.2f}px")
    print("  by noise:")
    for s, v in m['by_noise'].items():
        print(f"    {s:22s} n={v['n']:2d}  "
              f"<=5px {v['accuracy_curve']['5.0']*100:5.1f}%  "
              f"median {v['median_px']:.2f}px")
    tb = m['tie_break']
    print(f"  tie_break_used rate: {tb['rate']*100:.0f}%")
    for label in ('tie_break_used', 'resolved_outright'):
        if label in tb:
            v = tb[label]
            print(f"    {label:22s} n={v['n']:2d}  "
                  f"<=5px {v['accuracy_curve']['5.0']*100:5.1f}%  "
                  f"median {v['median_px']:.2f}px")
    print("  reject option (keep most-confident fraction):")
    for name, v in m['reject_option'].items():
        top = v['top_90pct']
        print(f"    {name:18s} rho(conf,err)={v['spearman_vs_error']:+.2f}  "
              f"top90%: <=5px {top['accuracy_curve']['5.0']*100:5.1f}% "
              f"median {top['median_px']:.2f}px")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', default='data/train')
    p.add_argument('--out', default='results/')
    p.add_argument('--no-sweep', action='store_true',
                   help='skip the lam/k calibration sweep')
    p.add_argument('--no-validate', action='store_true',
                   help='skip regenerating a fresh set to re-check constants')
    p.add_argument('--val-seed', type=int, default=4242)
    p.add_argument('--val-n', type=int, default=40)
    args = p.parse_args(argv)

    logging.disable(logging.WARNING)      # tie-break logging is Gate 7's job
    os.makedirs(args.out, exist_ok=True)
    records = load_pairs(args.data)
    print(f"Drift-Sense evaluation — {len(records)} pairs from {args.data}")

    print("\n[1/4] main evaluation with the shipped constants")
    rows = evaluate_set(args.data, records, keep_maps=True, progress=True)
    metrics = summarise(rows)
    from core.correlate import LAM
    from core.resolve import _K
    metrics['constants'] = {'lam': LAM, 'k': _K}
    print_report(metrics, "RESULTS (shipped constants "
                          f"lam={LAM}, k={_K})")

    plot_accuracy_curve(rows, metrics,
                        os.path.join(args.out, 'accuracy_curve.png'))
    plot_accuracy_vs_aperiodic(rows,
                               os.path.join(args.out,
                                            'accuracy_vs_aperiodic.png'))
    plot_confidence(rows, os.path.join(args.out, 'confidence_vs_error.png'))

    worst = max(rows, key=lambda r: r['error_px'])
    best = min(rows, key=lambda r: r['error_px'])
    plot_case(args.data, best, os.path.join(args.out, 'success_case.png'),
              f"SUCCESS — {best['id']} ({best['style']}), "
              f"error {best['error_px']:.2f} px")
    plot_case(args.data, worst, os.path.join(args.out, 'failure_case.png'),
              f"FAILURE (auto-selected worst of {len(rows)}) — {worst['id']} "
              f"({worst['style']}), error {worst['error_px']:.1f} px")
    metrics['success_case'] = {kk: best[kk] for kk in
                               ('id', 'style', 'aperiodic_level', 'error_px')}
    metrics['failure_case'] = {kk: worst[kk] for kk in
                               ('id', 'style', 'aperiodic_level', 'error_px',
                                'tie_break_used', 'family_size')}
    print(f"\n  auto-selected failure case: {worst['id']} "
          f"({worst['style']}, ap={worst['aperiodic_level']:.2f}) "
          f"error {worst['error_px']:.1f} px")

    if not args.no_sweep:
        ks = sorted(set(SWEEP_K + SWEEP_K_EXTRA))
        print(f"\n[2/4] calibration sweep: {len(SWEEP_LAM)} lam x {len(ks)} k "
              f"(brief's grid {SWEEP_K} extended down to {SWEEP_K_EXTRA})")
        table, bestc = sweep(args.data, records, args.out, SWEEP_LAM, ks)
        metrics['sweep'] = table
        metrics['sweep_best'] = bestc
        print(f"\n  best: lam={bestc['lam']}, k={bestc['k']} -> "
              f"acc@5px {bestc['acc_5px']*100:.1f}%, "
              f"median {bestc['median_px']:.2f}px")
        in_grid = [e for e in table if e['k'] in SWEEP_K]
        bg = max(in_grid, key=lambda e: (e['acc_5px'], -e['median_px']))
        metrics['sweep_best_within_brief_grid'] = bg
        print(f"  best within the brief's k grid only: lam={bg['lam']}, "
              f"k={bg['k']} -> acc@5px {bg['acc_5px']*100:.1f}%")
    else:
        bestc = None

    if not args.no_validate:
        val_dir = os.path.join(args.out, f'val_seed{args.val_seed}')
        print(f"\n[3/4] generating a FRESH {args.val_n}-pair set "
              f"(seed {args.val_seed}) to re-check the constants")
        if not os.path.exists(os.path.join(val_dir, 'manifest.json')):
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(
                                os.path.abspath(__file__)),
                                'generate_dataset.py'),
                            '--style', 'both', '--n', str(args.val_n),
                            '--out', val_dir, '--seed', str(args.val_seed)],
                           check=True, capture_output=True)
        vrecords = load_pairs(val_dir)
        print("\n[4/4] validating on the fresh set")
        vrows = evaluate_set(val_dir, vrecords)
        vmetrics = summarise(vrows)
        metrics['validation'] = {'seed': args.val_seed, 'dir': val_dir,
                                 **vmetrics}
        print_report(vmetrics, f"VALIDATION (fresh seed {args.val_seed}, "
                               f"shipped constants)")
        if bestc is not None:
            vrows2 = evaluate_set(val_dir, vrecords, lam=bestc['lam'],
                                  k=bestc['k'])
            e2 = np.array([r['error_px'] for r in vrows2])
            held = {'lam': bestc['lam'], 'k': bestc['k'],
                    'acc_5px_train': bestc['acc_5px'],
                    'acc_5px_val': acc_at(e2, 5.0),
                    'median_px_val': float(np.median(e2))}
            metrics['validation_of_swept_constants'] = held
            drop = held['acc_5px_train'] - held['acc_5px_val']
            print(f"\n  swept constants (lam={bestc['lam']}, k={bestc['k']}) "
                  f"on the fresh set: acc@5px "
                  f"{held['acc_5px_val']*100:.1f}% vs "
                  f"{held['acc_5px_train']*100:.1f}% on train "
                  f"(drop {drop*100:+.1f} pts)")
            print("  -> constants HOLD" if abs(drop) <= 0.10 else
                  "  -> WARNING: calibration did not transfer; treat the "
                  "swept constants as overfitted to one draw")

    for r in rows:
        r.pop('score_map', None)
    metrics['per_pair'] = [{kk: vv for kk, vv in r.items()
                            if kk not in ('score_map', 'diag')} for r in rows]
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1, default=str)
    print(f"\nwrote {args.out}: metrics.json, accuracy_curve.png, "
          f"accuracy_vs_aperiodic.png, confidence_vs_error.png, "
          f"success_case.png, failure_case.png, sweep.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
