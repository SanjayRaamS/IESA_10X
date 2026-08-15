"""Train the Drift-Sense confidence model (Phase 11).

    python train_confidence.py --data data/train --val results/val_seed4242

Fits a scikit-learn logistic regression that predicts CORRECT (error <= 5 px)
from six physics-normalised peak-surface features, and exports it as a ~2 KB
.npz that `core/confidence.py` reads with numpy alone.  scikit-learn is a
DEVELOPMENT dependency only (requirements-dev.txt); it is never imported at
runtime, so `localize.py`'s dependency set is unchanged.

This script is deliberately paranoid about overfitting, because 40 pairs with
~22 positives and 6 features is exactly the regime where a classifier looks
brilliant and means nothing:

  * stratified k-fold cross-validation on --data, with the standardiser fitted
    INSIDE each fold, so no test-fold statistics leak into training;
  * a single clean evaluation on --val, a set generated with a different seed
    that the model never sees during fitting or model selection;
  * a head-to-head against the existing `rescore_margin` reject option.  If the
    model does not beat that baseline it is not worth shipping, and this script
    says so in as many words.

Writes core/confidence_model.npz and results/confidence_report.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import evaluate as E
from core import confidence as C

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
KEEP_FRACTIONS = (1.0, 0.9, 0.75, 0.5)


def build_xy(data_dir, records, rows=None):
    """Feature matrix, labels and the baseline signal, for one dataset."""
    rows = rows if rows is not None else E.evaluate_set(data_dir, records)
    X = np.array([C.extract_features(r['diag']) for r in rows])
    y = np.array([r['error_px'] <= C.CORRECT_PX for r in rows], dtype=int)
    base = np.array([r['rescore_margin'] for r in rows], dtype=float)
    err = np.array([r['error_px'] for r in rows], dtype=float)
    return X, y, base, err, rows


def accuracy_at_keep(scoresig, err, frac):
    """Accuracy over the most-confident `frac` of predictions.

    `scoresig` is "higher == more confident".  This is the number a fab tool
    actually cares about: how good are the answers you choose to act on.
    """
    n = max(int(round(frac * len(err))), 1)
    keep = np.argsort(-np.asarray(scoresig, dtype=float))[:n]
    e = err[keep]
    return float(np.mean(e <= C.CORRECT_PX)), float(np.median(e)), n


def cross_val(X, y, Cval, seed=0, folds=5):
    """Stratified k-fold, standardiser fitted inside each fold.  Returns
    out-of-fold probabilities so every prediction is genuinely held out."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan)
    idx_by_class = {c: rng.permutation(np.nonzero(y == c)[0]) for c in (0, 1)}
    fold_of = np.empty(len(y), dtype=int)
    for c, idx in idx_by_class.items():
        fold_of[idx] = np.arange(len(idx)) % folds

    for f in range(folds):
        te = np.nonzero(fold_of == f)[0]
        tr = np.nonzero(fold_of != f)[0]
        if len(np.unique(y[tr])) < 2 or te.size == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        lr = LogisticRegression(C=Cval, max_iter=2000)
        lr.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof


def fit_final(X, y, Cval):
    from sklearn.linear_model import LogisticRegression

    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    lr = LogisticRegression(C=Cval, max_iter=2000)
    lr.fit((X - mu) / sd, y)
    return {'w': lr.coef_[0].astype(np.float64),
            'b': float(lr.intercept_[0]),
            'mu': mu.astype(np.float64), 'sd': sd.astype(np.float64),
            'features': C.FEATURES}


def report_block(name, sig, err, out):
    print(f"  {name}")
    rows = []
    for frac in KEEP_FRACTIONS:
        acc, med, n = accuracy_at_keep(sig, err, frac)
        rows.append({'keep': frac, 'n': n, 'acc_5px': acc, 'median_px': med})
        print(f"    keep {frac*100:5.1f}%  n={n:3d}  "
              f"<=5px {acc*100:5.1f}%  median {med:8.2f}px")
    rho = E._spearman(sig, err)
    print(f"    spearman(confidence, error) = {rho:+.2f}")
    out[name] = {'tiers': rows, 'spearman_vs_error': rho}
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', default='data/train')
    p.add_argument('--val', default='results/val_seed4242')
    p.add_argument('--out', default=C.DEFAULT_MODEL)
    p.add_argument('--report', default='results/confidence_report.json')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    try:
        import sklearn                                        # noqa: F401
    except ImportError:
        sys.stderr.write(
            "train_confidence.py needs scikit-learn (a DEV dependency).\n"
            f"  {sys.executable} -m pip install -r requirements-dev.txt\n"
            "Runtime is unaffected: core/confidence.py is numpy-only.\n")
        return 2

    print(f"[1/4] running the pipeline over {args.data}")
    recs = E.load_pairs(args.data)
    X, y, base, err, _ = build_xy(args.data, recs)
    print(f"      {len(y)} pairs, {int(y.sum())} correct / "
          f"{int((1 - y).sum())} incorrect, {X.shape[1]} features")

    print(f"\n[2/4] stratified {args.folds}-fold CV, selecting C")
    best = None
    for Cval in C_GRID:
        oof = cross_val(X, y, Cval, seed=args.seed, folds=args.folds)
        ok = np.isfinite(oof)
        acc = float(np.mean((oof[ok] >= 0.5) == y[ok]))
        auc = _auc(y[ok], oof[ok])
        print(f"      C={Cval:<5g} out-of-fold acc={acc*100:5.1f}%  "
              f"AUC={auc:.3f}")
        if best is None or auc > best['auc']:
            best = {'C': Cval, 'auc': auc, 'acc': acc, 'oof': oof}
    print(f"      chosen C={best['C']} (AUC {best['auc']:.3f})")

    out = {'n_train': int(len(y)), 'n_correct': int(y.sum()),
           'features': list(C.FEATURES), 'C': best['C'],
           'cv_folds': args.folds, 'cv_auc': best['auc'],
           'cv_accuracy': best['acc'], 'label': f'error <= {C.CORRECT_PX} px'}

    print(f"\n[3/4] reject option on {args.data} "
          f"(out-of-fold — never fitted on the pair it scores)")
    train_rep = {}
    report_block('logreg (out-of-fold)', best['oof'], err, train_rep)
    report_block('rescore_margin (baseline)', base, err, train_rep)
    out['train'] = train_rep

    model = fit_final(X, y, best['C'])
    np.savez(args.out, w=model['w'], b=model['b'], mu=model['mu'],
             sd=model['sd'], features=np.array(model['features']))
    size = os.path.getsize(args.out)
    print(f"\n      wrote {args.out} ({size} bytes)")
    order = sorted(zip(C.FEATURES, model['w']), key=lambda kv: -abs(kv[1]))
    print("      coefficients (standardised, largest |w| first):")
    for name, w in order:
        print(f"        {name:18s} {w:+.3f}")
    out['coefficients'] = {n: float(w) for n, w in zip(C.FEATURES, model['w'])}
    out['intercept'] = model['b']
    out['model_bytes'] = size

    if args.val and os.path.exists(os.path.join(args.val, 'manifest.json')):
        print(f"\n[4/4] HELD OUT: {args.val} (never seen in fitting or "
              f"model selection)")
        vrecs = E.load_pairs(args.val)
        Xv, yv, basev, errv, _ = build_xy(args.val, vrecs)
        pv = np.array([C.predict_proba(x, model) for x in Xv])
        acc_pt = np.asarray(pv >= 0.5)
        tp = int((acc_pt & (yv == 1)).sum())
        fp = int((acc_pt & (yv == 0)).sum())
        precision = tp / max(tp + fp, 1)
        print(f"      {len(yv)} pairs, {int(yv.sum())} correct")
        print(f"      accuracy={np.mean(acc_pt == yv)*100:.1f}%  "
              f"AUC={_auc(yv, pv):.3f}")
        print(f"      at the threshold-free p>=0.5 gate: accepts "
              f"{tp + fp}/{len(yv)}, of which {tp} are correct "
              f"(precision {precision*100:.1f}%)")
        out['val_gate'] = {'threshold': 0.5, 'accepted': tp + fp,
                           'accepted_correct': tp, 'precision': precision,
                           'n': int(len(yv))}
        val_rep = {}
        lr_tiers = report_block('logreg', pv, errv, val_rep)
        bl_tiers = report_block('rescore_margin (baseline)', basev, errv,
                                val_rep)
        out['val'] = {'dir': args.val, 'n': int(len(yv)),
                      'auc': _auc(yv, pv), **val_rep}

        # the verdict, stated plainly whichever way it falls
        wins = sum(a['acc_5px'] > b['acc_5px'] - 1e-12
                   for a, b in zip(lr_tiers, bl_tiers) if a['keep'] < 1.0)
        total = sum(1 for a in lr_tiers if a['keep'] < 1.0)
        # ">=", so a tie counts as passing.  Named for what it tests: the
        # model matching the baseline is not the model beating it.
        out['at_least_baseline_on_val'] = bool(wins >= total)
        out['strictly_beats_baseline_tiers'] = int(sum(
            a['acc_5px'] > b['acc_5px'] + 1e-12
            for a, b in zip(lr_tiers, bl_tiers) if a['keep'] < 1.0))
        print()
        if wins >= total:
            print(f"      VERDICT: logreg >= rescore_margin at all "
                  f"{total} reject thresholds on the held-out set.")
        else:
            print(f"      VERDICT: logreg beats the baseline at only "
                  f"{wins}/{total} thresholds on the held-out set. "
                  f"The single-feature reject option is not clearly "
                  f"improved on — say so on the slide.")

        fig_path = os.path.join(os.path.dirname(
            os.path.abspath(args.report)), 'confidence_model.png')
        ok = np.isfinite(best['oof'])          # mask y and p together
        plot_model([('train (out-of-fold)', y[ok], best['oof'][ok], '#3498db'),
                    ('held out (seed set)', yv, pv, '#2ecc71')],
                   ('held-out set', pv, errv), fig_path)
        print(f"      wrote {fig_path}")
    else:
        print(f"\n[4/4] skipped: no manifest at {args.val}. "
              f"Run evaluate.py first to generate the validation set.")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)),
                    exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(out, f, indent=1, default=float)
        print(f"\nwrote {args.report}")
    return 0


def _roc(y, p):
    """(fpr, tpr) sweeping every threshold.  n is small; brute force is fine."""
    y = np.asarray(y)
    thr = np.concatenate([[np.inf], np.sort(np.asarray(p))[::-1]])
    pos, neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tpr = [float(((p >= t) & (y == 1)).sum()) / pos for t in thr]
    fpr = [float(((p >= t) & (y == 0)).sum()) / neg for t in thr]
    return np.array(fpr), np.array(tpr)


def plot_model(curves, scatter, path):
    """Two panels: how well it separates, and how it behaves as a gate.

    curves  -- [(label, y, p, colour)] for the ROC panel
    scatter -- (label, p, err) for the held-out P(correct)-vs-error panel
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    axes[0].plot([0, 1], [0, 1], '--', color='#999', lw=1, label='chance')
    for label, y, p, colour in curves:
        fpr, tpr = _roc(y, p)
        axes[0].plot(fpr, tpr, '-', color=colour, lw=2,
                     label=f"{label} (AUC {_auc(y, p):.3f})")
    axes[0].set_xlabel('false positive rate')
    axes[0].set_ylabel('true positive rate')
    axes[0].set_title('ROC — predicting "error <= 5 px"')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)

    label, p, err = scatter
    e = np.maximum(np.asarray(err, dtype=float), 1e-3)   # log scale needs > 0
    good = e <= C.CORRECT_PX
    axes[1].scatter(np.asarray(p)[good], e[good], s=42, c='#2ecc71',
                    edgecolors='k', linewidths=0.4, label='correct', zorder=3)
    axes[1].scatter(np.asarray(p)[~good], e[~good], s=42, c='#e74c3c',
                    edgecolors='k', linewidths=0.4, label='incorrect', zorder=3)
    axes[1].axvline(0.5, color='#333', ls='--', lw=1.4,
                    label='accept threshold p = 0.5')
    axes[1].axhline(C.CORRECT_PX, color='#777', ls=':', lw=1.4,
                    label=f'correct <= {C.CORRECT_PX:g} px')
    axes[1].set_yscale('log')
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_xlabel('P(correct) from the logistic regression')
    axes[1].set_ylabel('actual error (px, log scale)')
    axes[1].set_title(f'P(correct) vs actual error — {label}')
    axes[1].legend(loc='center left', fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle('Drift-Sense confidence model — reject option only, '
                 'never the matcher', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _auc(y, p):
    """Rank AUC with tie handling.  Small n, so no need for anything clever."""
    y = np.asarray(y)
    pos, neg = np.asarray(p)[y == 1], np.asarray(p)[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    both = np.concatenate([pos, neg])
    for v in np.unique(both):             # average ranks within ties
        m = both == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


if __name__ == "__main__":
    sys.exit(main())
