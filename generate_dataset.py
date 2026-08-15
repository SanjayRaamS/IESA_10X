"""Standalone dataset generator for Drift-Sense.

    python generate_dataset.py --style dram|finfet|both --n 40 --out data/train --seed 0

Per pair:
  1. Render ONE large full-resolution ideal canvas of round(1000*scale) px
     square (scale in [9, 11] -> ~10000x10000 equivalent).  The canvas is
     float32 and core/layout.py renders it with chunked internals, so peak
     RAM stays ~1 GB.
  2. Pick the true centre uniformly in the search interior (80 px margin),
     rejection-sampled so it still respects the margin after the search
     image's own rotation jitter.
  3. REFERENCE = 1000x1000 full-mag crop around that location, imaged with
     image_sem(rng_ref).
  4. SEARCH = whole canvas INTER_AREA-downsampled to 1000x1000 (area
     averaging = a detector integrating a larger pixel footprint), imaged
     with image_sem(rng_search) at HIGHER noise: lower dose, higher readout,
     larger PSF — the brief says test search images are noisier.
  5. Reference and search each get their own small rotation and blur, so the
     recovered transform is non-trivial.  Ground truth is mapped through the
     search warp via sem.warp_matrix, so true_x/true_y are exact.

Outputs pair_NNNN_ref.png / pair_NNNN_search.png, manifest.json (list of
records) and generation_stats.json (timing, for slide 7).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from core.layout import render_dram, render_finfet
from core.sem import image_sem, warp_matrix

SEARCH_SIZE = 1000
REF_SIZE = 1000
MARGIN = 80


def sample_aperiodic(rng):
    """~20% of pairs near 0 (pure lattice — the hard cases), rest well away."""
    if rng.random() < 0.2:
        return float(rng.uniform(0.0, 0.05))
    return float(rng.uniform(0.3, 1.0))


def generate_pair(idx, style, master, aperiodic=None):
    """aperiodic: override the sampled aperiodic_level (the draw is still
    consumed, so every other random choice is unchanged)."""
    layout_seed, ref_seed, search_seed = (int(v) for v in
                                          master.integers(2 ** 31, size=3))
    s = float(master.uniform(9.0, 11.0))          # true magnification ratio
    rot_ref = float(master.uniform(-3.0, 3.0))    # independent per-image jitter
    rot_search = float(master.uniform(-3.0, 3.0))
    ap = sample_aperiodic(master)
    if aperiodic is not None:
        ap = float(aperiodic)
    ref_params = {'edge_gain': float(master.uniform(0.25, 0.45)),
                  'rotation_deg': rot_ref, 'scale': 1.0,
                  'psf_sigma': float(master.uniform(1.0, 1.6)),
                  'shading': float(master.uniform(0.04, 0.12)),
                  'dose': float(master.uniform(200.0, 400.0)),
                  'readout': float(master.uniform(0.01, 0.02))}
    # Search PSF is ~1 px in ITS OWN pixel units: the physical beam spot is
    # nm-scale and constant, so at ~10x lower mag it shrinks 10x in pixels and
    # the low-mag blur is detector/raster-limited.  In PHYSICAL units this is
    # still ~10x the reference PSF.  The noisier-search asymmetry the brief
    # demands lives in dose and readout.  (psf_sigma >= 1.4 search px would
    # annihilate fine-pitch lattices entirely — measured in Gate 3 dev.)
    search_params = {'edge_gain': float(master.uniform(0.25, 0.45)),
                     'rotation_deg': rot_search, 'scale': 1.0,
                     'psf_sigma': float(master.uniform(0.9, 1.25)),
                     'shading': float(master.uniform(0.04, 0.12)),
                     'dose': float(master.uniform(90.0, 160.0)),
                     'readout': float(master.uniform(0.02, 0.035))}

    N = int(round(SEARCH_SIZE * s))
    layout_params = {'aperiodic_level': ap}
    if style == 'dram':
        # keep the lattice resolvable at search magnification (>= ~4.4 px per
        # period after downscale) — an operator picks the search mag so the
        # pattern is still visible
        pmin = max(30.0, 4.4 * s)
        layout_params['pitch_x'] = float(master.uniform(pmin, 60.0))
        layout_params['pitch_y'] = float(master.uniform(pmin, 60.0))
    render = render_dram if style == 'dram' else render_finfet
    canvas, meta = render((N, N), np.random.default_rng(layout_seed),
                          layout_params)

    # true centre: uniform in the interior, kept 80 px in-bounds after the
    # search image's own rotation moves the content
    M_search, _ = warp_matrix((SEARCH_SIZE, SEARCH_SIZE), rot_search, 1.0)
    for _ in range(100):
        cx, cy = master.uniform(MARGIN + 50, SEARCH_SIZE - MARGIN - 50, size=2)
        x0 = int(round((cx + 0.5) * s - REF_SIZE / 2.0))
        y0 = int(round((cy + 0.5) * s - REF_SIZE / 2.0))
        if not (0 <= x0 <= N - REF_SIZE and 0 <= y0 <= N - REF_SIZE):
            continue
        # crop centre in physical canvas units -> search px -> through warp
        sx = (x0 + REF_SIZE / 2.0) / s - 0.5
        sy = (y0 + REF_SIZE / 2.0) / s - 0.5
        tx, ty = M_search @ np.array([sx, sy, 1.0])
        if not (MARGIN <= tx <= SEARCH_SIZE - MARGIN and
                MARGIN <= ty <= SEARCH_SIZE - MARGIN):
            continue
        # a reference must contain structure: reject crops that landed in a
        # blank field / array-boundary region (no operator would navigate
        # from a featureless reference)
        if float(canvas[y0:y0 + REF_SIZE, x0:x0 + REF_SIZE].std()) > 0.08:
            break
    else:
        raise RuntimeError(f"pair {idx}: could not place true centre in-bounds")

    ref_ideal = canvas[y0:y0 + REF_SIZE, x0:x0 + REF_SIZE]
    ref = image_sem(ref_ideal, np.random.default_rng(ref_seed), ref_params)
    search_ideal = cv2.resize(canvas, (SEARCH_SIZE, SEARCH_SIZE),
                              interpolation=cv2.INTER_AREA)
    search = image_sem(search_ideal, np.random.default_rng(search_seed),
                       search_params)

    rec = {'id': f'pair_{idx:04d}', 'style': style,
           'ref_path': f'pair_{idx:04d}_ref.png',
           'search_path': f'pair_{idx:04d}_search.png',
           'true_x': float(tx), 'true_y': float(ty),
           'scale': s, 'rotation_deg': rot_search - rot_ref,
           'aperiodic_level': ap, 'search_dose': search_params['dose'],
           'ref_seed': ref_seed, 'search_seed': search_seed,
           'pitch_x': float(meta['pitch_x']), 'pitch_y': float(meta['pitch_y']),
           # extras (not required by the manifest spec, used by tests/tools)
           'layout_seed': layout_seed,
           'ref_rotation_deg': rot_ref, 'search_rotation_deg': rot_search,
           'ref_dose': ref_params['dose']}
    return ref, search, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--style', choices=['dram', 'finfet', 'both'],
                    default='both')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--out', default='data/train')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aperiodic', type=float, default=None,
                    help='force aperiodic_level (e.g. 0 for pure lattices)')
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    master = np.random.default_rng(args.seed)
    records = []
    t0 = time.time()
    for i in range(args.n):
        style = args.style if args.style != 'both' else \
            str(master.choice(['dram', 'finfet']))
        tp = time.time()
        ref, search, rec = generate_pair(i, style, master, args.aperiodic)
        cv2.imwrite(os.path.join(args.out, rec['ref_path']), ref)
        cv2.imwrite(os.path.join(args.out, rec['search_path']), search)
        records.append(rec)
        print(f"{rec['id']} {style:6s} scale={rec['scale']:.2f} "
              f"ap={rec['aperiodic_level']:.2f} "
              f"true=({rec['true_x']:.1f},{rec['true_y']:.1f}) "
              f"[{time.time() - tp:.1f}s]", flush=True)
    total = time.time() - t0

    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(records, f, indent=1)
    stats = {'n': args.n, 'style': args.style, 'seed': args.seed,
             'aperiodic': args.aperiodic,
             'total_seconds': round(total, 2),
             'seconds_per_pair': round(total / max(args.n, 1), 2)}
    with open(os.path.join(args.out, 'generation_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {args.n} pairs to {args.out} in {total:.1f}s "
          f"({stats['seconds_per_pair']:.1f}s/pair)")


if __name__ == '__main__':
    main()
