"""Verify Gate 7 for core/resolve.py.

  a) CONSTRUCTED: a hand-built score map with two EQUAL peaks at (300,300)
     and (520,510) in a (1000,1000) search image must resolve to (520,510) --
     the one nearer the centre -- with tie_break_used True.  A third case with
     equal AND equidistant peaks must be deterministic over ten runs.
  b) 10 pairs at aperiodic_level=0: tie_break_used must fire on most of them,
     and accuracy must be materially better than random peak selection.
     Both numbers are printed.

Run: python tests/test_resolve.py   (prints PASS on success)
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.correlate import build_template, combine, match_psf, score_maps
from core.lattice import estimate_transform
from core.preprocess import anscombe, normalise, prep
from core.resolve import _local_maxima, resolve

AMB = os.path.join(ROOT, 'data', 'ambiguous')
N_AMB = 10


def _blob(S, x, y, amp=1.0, sigma=3.0):
    h, w = S.shape
    y0, y1 = max(int(y) - 20, 0), min(int(y) + 21, h)
    x0, x1 = max(int(x) - 20, 0), min(int(x) + 21, w)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    S[y0:y1, x0:x1] += amp * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))


def gate_a():
    # smooth background noise so the robust sigma is a realistic non-zero
    rng = np.random.default_rng(0)
    S = cv2.GaussianBlur(rng.normal(0, 0.01, (1000, 1000)).astype(np.float32),
                         (0, 0), 2.0)
    _blob(S, 300, 300)
    _blob(S, 520, 510)
    S[300, 300] = S[510, 520] = 1.0            # exactly equal height
    x, y, info = resolve(S, (1000, 1000), pitch=10.0)
    d_near = np.hypot(520 - 500, 510 - 500)
    d_far = np.hypot(300 - 500, 300 - 500)
    assert (x, y) == (520.0, 510.0), f"centre rule picked ({x},{y}), want (520,510)"
    assert info['tie_break_used'] is True, "tie_break_used should be True"
    assert info['family_size'] >= 2, info['family_size']
    print(f"  (a) equal peaks at (300,300) d={d_far:.1f} and (520,510) "
          f"d={d_near:.1f} -> ({x:.0f},{y:.0f}), tie_break_used="
          f"{info['tie_break_used']}, family={info['family_size']}")

    # equal AND equidistant: must be deterministic
    S2 = cv2.GaussianBlur(rng.normal(0, 0.01, (1000, 1000)).astype(np.float32),
                          (0, 0), 2.0)
    _blob(S2, 400, 500)
    _blob(S2, 600, 500)
    S2[500, 400] = S2[500, 600] = 1.0
    outs = {resolve(S2, (1000, 1000), pitch=10.0)[:2] for _ in range(10)}
    assert len(outs) == 1, f"non-deterministic on equidistant ties: {outs}"
    (xd, yd) = outs.pop()
    dd = np.hypot(xd - 500, yd - 500)
    print(f"  (a) equidistant ties (both d={dd:.1f}) -> ({xd:.0f},{yd:.0f}) "
          f"identical on 10/10 runs")


def ensure_ambiguous():
    man = os.path.join(AMB, 'manifest.json')
    if not os.path.exists(man):
        subprocess.run([sys.executable,
                        os.path.join(ROOT, 'generate_dataset.py'),
                        '--style', 'both', '--n', str(N_AMB),
                        '--out', AMB, '--seed', '7',
                        '--aperiodic', '0'], check=True)
    with open(man) as f:
        return json.load(f)


def gate_b():
    records = ensure_ambiguous()
    rng = np.random.default_rng(12345)
    n_tie, res_err, rnd_err = 0, [], []
    for rec in records:
        refu = cv2.imread(os.path.join(AMB, rec['ref_path']),
                          cv2.IMREAD_GRAYSCALE)
        seau = cv2.imread(os.path.join(AMB, rec['search_path']),
                          cv2.IMREAD_GRAYSCALE)
        search = prep(seau)
        A, _ = estimate_transform(prep(refu), search, scale_prior=0.1)
        tmpl = match_psf(normalise(build_template(anscombe(refu), A)), search)
        S = combine(*score_maps(tmpl, search))
        x, y, info = resolve(S, search.shape)
        n_tie += info['tie_break_used']
        res_err.append(np.hypot(x - rec['true_x'], y - rec['true_y']))

        # baseline: pick a RANDOM member of the same tied family
        xs, ys, sc = _local_maxima(S, np.isfinite(S),
                                   0.7 * info['pitch'])
        fam = np.nonzero((sc[0] - sc) < 2.0 * info['sigma'])[0]
        j = int(rng.integers(fam.size))
        rnd_err.append(np.hypot(xs[fam[j]] - rec['true_x'],
                                ys[fam[j]] - rec['true_y']))
    res_err, rnd_err = np.asarray(res_err), np.asarray(rnd_err)
    print(f"  (b) {len(records)} pairs at aperiodic_level=0:")
    print(f"        tie_break_used fired on {n_tie}/{len(records)}")
    print(f"        centre rule    : median err {np.median(res_err):7.1f} px  "
          f"mean {res_err.mean():7.1f} px")
    print(f"        random-in-family: median err {np.median(rnd_err):7.1f} px  "
          f"mean {rnd_err.mean():7.1f} px")
    assert n_tie > len(records) / 2, \
        f"tie_break_used fired only {n_tie}/{len(records)} times"
    assert res_err.mean() < rnd_err.mean(), \
        f"centre rule ({res_err.mean():.1f}) not better than random " \
        f"({rnd_err.mean():.1f})"
    print(f"        centre rule is {100 * (1 - res_err.mean() / rnd_err.mean()):.0f}%"
          f" better than random peak selection")


if __name__ == "__main__":
    print("Verify Gate 7 (core/resolve.py)")
    gate_a()
    gate_b()
    print("PASS")
