"""Verify Gate 8 for core/refine.py.

On the 15 pairs with the LOWEST noise (highest search dose), run the full
pipeline — lattice geometry, dual-channel scoring, ambiguity resolution,
subpixel refinement — and assert

    median localisation error < 0.5 px
    90th percentile          < 1.5 px

The error histogram is printed either way.

Run: python tests/test_refine.py   (prints PASS on success)
"""

import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.correlate import build_template, combine, match_psf, score_maps
from core.lattice import estimate_transform
from core.preprocess import anscombe, normalise, prep
from core.refine import refine, rescore_candidates
from core.resolve import resolve

OUT = os.path.join(ROOT, 'data', 'train')
N_LOW_NOISE = 15
BINS = [0.0, 0.25, 0.5, 1.0, 1.5, 5.0, 50.0, np.inf]


def locate(rec):
    refu = cv2.imread(os.path.join(OUT, rec['ref_path']), cv2.IMREAD_GRAYSCALE)
    seau = cv2.imread(os.path.join(OUT, rec['search_path']),
                      cv2.IMREAD_GRAYSCALE)
    search = prep(seau)
    A, _ = estimate_transform(prep(refu), search, scale_prior=0.1)
    tmpl = match_psf(normalise(build_template(anscombe(refu), A)), search)
    S = combine(*score_maps(tmpl, search))
    px, py, rinfo = resolve(S, search.shape)
    # re-rank the top peaks with every pixel before refining: subpixel
    # refinement cannot rescue a wrong unit cell (see rescore_candidates)
    (bx, by), _, sinfo = rescore_candidates(S, anscombe(refu), search, A,
                                            template=tmpl,
                                            pitch=rinfo['pitch'])
    x, y, dinfo = refine(S, (bx, by), anscombe(refu), search, A,
                         template=tmpl)
    dinfo.update(sinfo)
    return (px, py), (x, y), rinfo, dinfo


def histogram(errs):
    print("      error histogram (px):")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        n = int(((errs >= lo) & (errs < hi)).sum())
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"        {label:>10s} | {'#' * n:<20s} {n}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)      # tie-break logging is Phase 7's gate
    print("Verify Gate 8 (core/refine.py)")
    with open(os.path.join(OUT, 'manifest.json')) as f:
        records = json.load(f)
    # lowest noise == highest electron dose in the search capture
    low = sorted(records, key=lambda r: -r['search_dose'])[:N_LOW_NOISE]
    print(f"  {N_LOW_NOISE} lowest-noise pairs "
          f"(search_dose {low[-1]['search_dose']:.0f}-{low[0]['search_dose']:.0f})")

    errs, coarse_errs, n_ecc, n_ecc_rej, n_par_rej = [], [], 0, 0, 0
    for rec in low:
        (px, py), (x, y), rinfo, dinfo = locate(rec)
        errs.append(np.hypot(x - rec['true_x'], y - rec['true_y']))
        coarse_errs.append(np.hypot(px - rec['true_x'], py - rec['true_y']))
        n_ecc += dinfo['method'] == 'ecc'
        n_ecc_rej += dinfo['ecc_rejected']
        n_par_rej += dinfo['parabolic_rejected']
    errs = np.asarray(errs)
    coarse = np.asarray(coarse_errs)

    print(f"  ECC accepted on {n_ecc}/{len(low)}, ECC rejected by guard "
          f"{n_ecc_rej}, parabolic rejected {n_par_rej}")
    print(f"  integer peak : median {np.median(coarse):8.2f} px  "
          f"p90 {np.percentile(coarse, 90):8.2f} px")
    print(f"  refined      : median {np.median(errs):8.2f} px  "
          f"p90 {np.percentile(errs, 90):8.2f} px")
    histogram(errs)

    # subpixel accuracy is only meaningful where the correct cell was chosen
    good = errs[errs < 15.0]
    if good.size:
        print(f"  among the {good.size}/{len(low)} pairs on the correct cell: "
              f"median {np.median(good):.3f} px, p90 "
              f"{np.percentile(good, 90):.3f} px")

    med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    assert med < 0.5, f"median localisation error {med:.3f} px >= 0.5"
    assert p90 < 1.5, f"p90 localisation error {p90:.3f} px >= 1.5"
    print(f"  median {med:.3f} px < 0.5 | p90 {p90:.3f} px < 1.5")
    print("PASS")
