"""Verify Gate 11 for core/confidence.py — the reject option.

  a) The shipped model loads, matches FEATURES, and is small enough to commit.
  b) A decisive score surface must rank above an ambiguous one.
  c) localize.py must degrade silently when the model is missing or corrupt —
     same coordinate, no traceback, no stderr noise.
  d) On the held-out set the model must separate correct from incorrect well
     above chance (AUC >= 0.80), so a broken retrain cannot ship quietly.
  e) The model must never change the answer: coordinates with and without it
     must be bit-identical.

Each check prints PASS.  (d) is skipped with a printed notice if the held-out
set has not been generated yet (run evaluate.py first).

Run: python tests/test_confidence.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import confidence as C            # noqa: E402

VAL = os.path.join(ROOT, 'results', 'val_seed4242')
PAIR = (os.path.join(ROOT, 'data', 'train', 'pair_0000_ref.png'),
        os.path.join(ROOT, 'data', 'train', 'pair_0000_search.png'))

DECISIVE = {'resolve': {'best_score': 1.0, 'second_score': 0.2,
                        'confidence': 8.0, 'family_size': 1},
            'rescore': {'best_residual_zncc': 0.5, 'best_ecc_cc': 0.95},
            'transform': {'sharpness': 2.0}}
AMBIGUOUS = {'resolve': {'best_score': 1.0, 'second_score': 0.99,
                         'confidence': 0.1, 'family_size': 6},
             'rescore': {'best_residual_zncc': 0.02, 'best_ecc_cc': 0.55},
             'transform': {'sharpness': 2.0}}


def _localize(cwd=None):
    """Run the CLI and return its stdout coordinate line."""
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'localize.py'),
                          '--ref', PAIR[0], '--search', PAIR[1]],
                         capture_output=True, text=True, cwd=cwd or ROOT)
    assert out.returncode == 0, f"localize.py exited {out.returncode}"
    return out.stdout.strip().splitlines()[-1], out.stderr


def gate_a():
    print("  (a) model loads and is committable")
    m = C.load_model()
    assert m is not None, f"no model at {C.DEFAULT_MODEL} — run train_confidence.py"
    assert m['features'] == C.FEATURES, m['features']
    assert m['w'].shape == (len(C.FEATURES),), m['w'].shape
    assert m['mu'].shape == m['sd'].shape == m['w'].shape
    assert np.all(m['sd'] > 0), "degenerate standardiser"
    size = os.path.getsize(C.DEFAULT_MODEL)
    assert size < 16384, f"model is {size} bytes; expected a few KB"
    print(f"      {len(C.FEATURES)} features, {size} bytes")
    print("      PASS")
    return m


def gate_b(m):
    print("  (b) decisive surface outranks ambiguous")
    pd, pa = C.score(DECISIVE, m), C.score(AMBIGUOUS, m)
    assert pd > pa, f"decisive {pd:.3f} !> ambiguous {pa:.3f}"
    assert 0.0 <= pa <= 1.0 and 0.0 <= pd <= 1.0, (pd, pa)
    print(f"      decisive {pd:.3f} > ambiguous {pa:.3f}")
    print("      PASS")


def gate_c():
    print("  (c) degrades silently without a usable model")
    baseline, _ = _localize()
    tmp = tempfile.mkdtemp(prefix='driftsense_gate11_')
    backup = os.path.join(tmp, 'model.npz')
    shutil.copy2(C.DEFAULT_MODEL, backup)
    try:
        os.remove(C.DEFAULT_MODEL)
        missing, err_missing = _localize()
        assert missing == baseline, f"{missing!r} != {baseline!r}"
        assert 'Traceback' not in err_missing, err_missing

        with open(C.DEFAULT_MODEL, 'wb') as f:
            f.write(os.urandom(256))            # not a valid .npz
        corrupt, err_corrupt = _localize()
        assert corrupt == baseline, f"{corrupt!r} != {baseline!r}"
        assert 'Traceback' not in err_corrupt, err_corrupt
    finally:
        shutil.copy2(backup, C.DEFAULT_MODEL)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"      missing and corrupt both -> {baseline}")
    print("      PASS")
    return baseline


def gate_d(m):
    print("  (d) held-out separation")
    if not os.path.exists(os.path.join(VAL, 'manifest.json')):
        print(f"      SKIPPED: no manifest at {VAL} (run evaluate.py first)")
        return
    import evaluate as E
    import train_confidence as T

    recs = E.load_pairs(VAL)
    X, y, _base, _err, _rows = T.build_xy(VAL, recs)
    p = np.array([C.predict_proba(x, m) for x in X])
    auc = T._auc(y, p)
    acc = float(np.mean((p >= 0.5) == y))
    assert auc >= 0.80, f"held-out AUC {auc:.3f} < 0.80"
    print(f"      n={len(y)}  AUC={auc:.3f}  accuracy@0.5={acc*100:.1f}%")
    print("      PASS")


def gate_e(baseline):
    print("  (e) the model never moves the answer")
    with_model, _ = _localize()
    assert with_model == baseline, f"{with_model!r} != {baseline!r}"
    print(f"      {with_model} with and without the model")
    print("      PASS")


if __name__ == "__main__":
    print("Verify Gate 11 (core/confidence.py)")
    model = gate_a()
    gate_b(model)
    coord = gate_c()
    gate_d(model)
    gate_e(coord)
    print("PASS")
