"""Verify Gate 6 for core/correlate.py.

  a) On the 8 pairs with the HIGHEST aperiodic_level, argmax(S) must be within
     15 px of ground truth for at least 7 of 8.
  b) On pairs with aperiodic_level near 0, S_full must have MULTIPLE peaks
     within 2% of its max (proving the ambiguity is real, not a bug), while
     S_res is comparatively flat.  Both peak counts are printed.

Run: python tests/test_correlate.py   (prints PASS on success)
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.correlate import (build_template, combine, match_psf, score_maps)
from core.lattice import estimate_transform
from core.preprocess import anscombe, normalise, prep

OUT = os.path.join(ROOT, 'data', 'train')


def pipeline(rec):
    refu = cv2.imread(os.path.join(OUT, rec['ref_path']), cv2.IMREAD_GRAYSCALE)
    seau = cv2.imread(os.path.join(OUT, rec['search_path']), cv2.IMREAD_GRAYSCALE)
    search = prep(seau)
    A, diag = estimate_transform(prep(refu), search, scale_prior=0.1)
    # normalise AFTER the warp so flattening/LCN act at the same physical
    # scale in both images (see core.preprocess.normalise)
    tmpl = match_psf(normalise(build_template(anscombe(refu), A)), search)
    S_full, S_res = score_maps(tmpl, search)
    return S_full, S_res, combine(S_full, S_res), diag


def argmax_xy(M):
    finite = np.isfinite(M)
    y, x = np.unravel_index(int(np.nanargmax(np.where(finite, M, -np.inf))),
                            M.shape)
    return float(x), float(y)


def count_near_max_peaks(M, frac=0.02, min_sep=15):
    """Distinct local maxima within `frac` of the map maximum."""
    finite = np.isfinite(M)
    top = float(M[finite].max())
    thr = top - frac * abs(top)
    k = 2 * min_sep + 1
    mx = cv2.dilate(np.where(finite, M, -np.inf).astype(np.float32),
                    np.ones((k, k), np.float32))
    hits = finite & (M >= thr) & (M >= mx - 1e-9)
    n, _, _, _ = cv2.connectedComponentsWithStats(hits.astype(np.uint8), 8)
    return n - 1, top


def load():
    with open(os.path.join(OUT, 'manifest.json')) as f:
        return json.load(f)


def gate_a(records, results):
    hi = sorted(range(len(records)),
                key=lambda i: -records[i]['aperiodic_level'])[:8]
    ok = 0
    print("  (a) 8 highest-aperiodic pairs:")
    for i in hi:
        rec = records[i]
        x, y = argmax_xy(results[i][2])
        err = float(np.hypot(x - rec['true_x'], y - rec['true_y']))
        ok += err < 15.0
        print(f"        {rec['id']} {rec['style']:6s} ap={rec['aperiodic_level']:.2f}"
              f"  err={err:8.1f} px {'ok' if err < 15 else 'MISS'}")
    print(f"      {ok}/8 within 15 px (need >= 7)")
    assert ok >= 7, f"only {ok}/8 within 15 px"


def gate_b(records, results):
    lo = [i for i in range(len(records))
          if records[i]['aperiodic_level'] < 0.06]
    assert lo, "no near-zero aperiodic_level pairs in the manifest"
    nfull, nres = [], []
    print("  (b) near-zero aperiodic_level pairs (ambiguity should be real):")
    for i in lo:
        S_full, S_res = results[i][0], results[i][1]
        cf, tf = count_near_max_peaks(S_full)
        cr, tr = count_near_max_peaks(S_res)
        nfull.append(cf)
        nres.append(cr)
        # flatness: how far the top peak stands out of the map's own spread
        ff = (tf - np.nanmean(S_full[np.isfinite(S_full)])) / \
            (np.nanstd(S_full[np.isfinite(S_full)]) + 1e-12)
        fr = (tr - np.nanmean(S_res[np.isfinite(S_res)])) / \
            (np.nanstd(S_res[np.isfinite(S_res)]) + 1e-12)
        print(f"        {records[i]['id']} ap={records[i]['aperiodic_level']:.3f}"
              f"  S_full peaks within 2% = {cf:4d} (peak {ff:5.1f} sigma) |"
              f"  S_res peaks = {cr:4d} (peak {fr:5.1f} sigma)")
    print(f"      median S_full near-max peaks = {np.median(nfull):.0f}, "
          f"S_res = {np.median(nres):.0f}")
    assert np.median(nfull) > 1, (
        f"S_full median near-max peak count {np.median(nfull)} is not >1: "
        "the periodic ambiguity should be REAL at aperiodic_level~0")


if __name__ == "__main__":
    print("Verify Gate 6 (core/correlate.py)")
    records = load()
    results = [pipeline(r) for r in records]
    gate_b(records, results)
    gate_a(records, results)
    print("PASS")
