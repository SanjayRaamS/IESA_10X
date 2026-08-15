"""Verify Gate 4 for core/preprocess.py.

Repeats the Gate 2(a) variance-vs-mean measurement on flat patches, before
and after prep().  The normalised (dimensionless) slope must drop by >5x,
proving the Anscombe VST + LCN removed the signal-dependence that would bias
ZNCC.  Slopes are printed.

Run: python tests/test_preprocess.py   (prints PASS on success)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from core.preprocess import prep
from core.sem import image_sem

CROP = 40  # discard LCN/flattening window edge effects


def slopes():
    levels = np.linspace(0.05, 0.85, 10)
    p = {'edge_gain': 0.0, 'rotation_deg': 0.0, 'scale': 1.0, 'psf_sigma': 1.0,
         'shading': 0.0, 'dose': 200.0, 'readout': 0.01}
    m_raw, v_raw, v_prep = [], [], []
    for i, lv in enumerate(levels):
        ideal = np.full((256, 256), lv, dtype=np.float32)
        raw = image_sem(ideal, np.random.default_rng(10 + i), p)
        pre = prep(raw)[CROP:-CROP, CROP:-CROP]
        raw = raw.astype(np.float64) / 255.0
        m_raw.append(raw.mean())
        v_raw.append(raw.var())
        v_prep.append(float(pre.var()))
    m_raw, v_raw, v_prep = map(np.asarray, (m_raw, v_raw, v_prep))
    s_before = np.polyfit(m_raw, v_raw, 1)[0]
    s_after = np.polyfit(m_raw, v_prep, 1)[0]
    # dimensionless (elasticity) form so the two slopes are comparable:
    # d(var)/d(mean) * mean/var — relative variance change per relative
    # mean change, independent of each signal's units
    e_before = s_before * m_raw.mean() / v_raw.mean()
    e_after = s_after * m_raw.mean() / v_prep.mean()
    return s_before, s_after, e_before, e_after


if __name__ == "__main__":
    print("Verify Gate 4 (core/preprocess.py)")
    s_b, s_a, e_b, e_a = slopes()
    print(f"  raw:     var-vs-mean slope {s_b:+.3e}  (normalised {e_b:+.3f})")
    print(f"  prep():  var-vs-mean slope {s_a:+.3e}  (normalised {e_a:+.3f})")
    ratio = abs(e_b) / max(abs(e_a), 1e-12)
    assert ratio > 5.0, f"normalised slope only dropped {ratio:.1f}x (need >5x)"
    print(f"  signal-dependence reduced {ratio:.0f}x (> 5x)")
    print("PASS")
