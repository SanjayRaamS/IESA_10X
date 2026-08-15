"""Verify Gate 5 for core/lattice.py.

Runs estimate_transform on all 40 generated pairs and compares the recovered
scale and rotation against manifest ground truth.  Asserts:
    median |rotation error| < 0.2 deg
    median |scale error| / scale < 0.5%
    95th percentile rotation error < 0.6 deg
Prints the full error distribution (needed for the slides).

Run: python tests/test_lattice.py   (prints PASS on success)
"""

import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.lattice import estimate_transform
from core.preprocess import prep

OUT = os.path.join(ROOT, 'data', 'train')


def run():
    with open(os.path.join(OUT, 'manifest.json')) as f:
        records = json.load(f)
    rows = []
    t0 = time.time()
    for rec in records:
        ref = cv2.imread(os.path.join(OUT, rec['ref_path']), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(OUT, rec['search_path']),
                            cv2.IMREAD_GRAYSCALE)
        A, d = estimate_transform(prep(ref), prep(search), scale_prior=0.1)
        # manifest 'scale' is the magnification RATIO (search = ref / scale)
        true_scale = 1.0 / rec['scale']
        true_rot = rec['rotation_deg']
        rows.append({
            'id': rec['id'], 'style': rec['style'], 'method': d['method'],
            'scale_err': abs(d['scale'] - true_scale) / true_scale,
            'rot_err': abs(d['rotation_deg'] - true_rot),
            'aniso': d['anisotropy'], 'n_matched': d.get('n_matched', 0),
        })
    return rows, time.time() - t0


def percentiles(v, label, unit, scale=1.0):
    v = np.asarray(v) * scale
    print(f"  {label:22s} median={np.median(v):7.4f}  p75={np.percentile(v, 75):7.4f}"
          f"  p95={np.percentile(v, 95):7.4f}  max={v.max():7.4f} {unit}")
    return v


if __name__ == "__main__":
    print("Verify Gate 5 (core/lattice.py)")
    rows, elapsed = run()
    n = len(rows)
    rot = np.array([r['rot_err'] for r in rows])
    sca = np.array([r['scale_err'] for r in rows])

    print(f"\n  {n} pairs in {elapsed:.1f}s ({1000 * elapsed / n:.0f} ms/pair)")
    print(f"  methods: {dict(Counter(r['method'] for r in rows))}")
    print(f"  peaks matched: median={np.median([r['n_matched'] for r in rows]):.0f}"
          f"  min={min(r['n_matched'] for r in rows)}")
    print()
    percentiles(rot, "rotation error", "deg")
    percentiles(sca, "scale error", "%", 100.0)
    for style in sorted({r['style'] for r in rows}):
        sub = [r for r in rows if r['style'] == style]
        sr = np.array([r['rot_err'] for r in sub])
        ss = np.array([r['scale_err'] for r in sub]) * 100.0
        print(f"  {style:6s} (n={len(sub):2d})        rot median={np.median(sr):.4f} "
              f"p95={np.percentile(sr, 95):.4f} deg | scale median={np.median(ss):.4f}"
              f" p95={np.percentile(ss, 95):.4f} %")

    worst = sorted(rows, key=lambda r: -r['rot_err'])[:3]
    print("\n  worst rotation errors: " + ", ".join(
        f"{r['id']}({r['style']},{r['rot_err']:.3f}deg,nm={r['n_matched']})"
        for r in worst))

    med_rot, p95_rot, med_sca = (float(np.median(rot)),
                                 float(np.percentile(rot, 95)),
                                 float(np.median(sca)))
    print()
    assert med_rot < 0.2, f"median rotation error {med_rot:.4f} deg >= 0.2"
    assert med_sca < 0.005, f"median scale error {med_sca * 100:.4f}% >= 0.5%"
    assert p95_rot < 0.6, f"p95 rotation error {p95_rot:.4f} deg >= 0.6"
    print(f"  median rot {med_rot:.4f} deg < 0.2 | median scale "
          f"{med_sca * 100:.4f}% < 0.5% | p95 rot {p95_rot:.4f} deg < 0.6")
    print("PASS")
