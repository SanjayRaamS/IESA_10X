"""Verify Gate 3 for generate_dataset.py.

  a) Round-trip: crop the search at (true_x, true_y), upscale, ZNCC vs the
     reference > 0.5.  If this fails the ground truth is wrong and everything
     downstream measures noise.
  b) Every manifest record has all required keys; true coords respect the
     80 px margin.
  c) Ref/search PNGs are not byte-identical and their noise fields are
     uncorrelated (catches any reuse of a noise array between captures).
  d) Reports total generation time for 40 pairs (slide 7).

Generates data/train (40 pairs, seed 0) via the CLI if not already present.
Run: python tests/test_pairs.py   (prints PASS on success)
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

OUT = os.path.join(ROOT, 'data', 'train')
MARGIN = 80
REQUIRED_KEYS = ['id', 'style', 'ref_path', 'search_path', 'true_x', 'true_y',
                 'scale', 'rotation_deg', 'aperiodic_level', 'search_dose',
                 'ref_seed', 'search_seed', 'pitch_x', 'pitch_y']


def ensure_dataset():
    man = os.path.join(OUT, 'manifest.json')
    stats = os.path.join(OUT, 'generation_stats.json')
    if not (os.path.exists(man) and os.path.exists(stats)):
        subprocess.run([sys.executable,
                        os.path.join(ROOT, 'generate_dataset.py'),
                        '--style', 'both', '--n', '40',
                        '--out', OUT, '--seed', '0'], check=True)
    with open(man) as f:
        records = json.load(f)
    with open(stats) as f:
        st = json.load(f)
    return records, st


def _load(rec, which):
    img = cv2.imread(os.path.join(OUT, rec[which]), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (1000, 1000), rec['id']
    return img


def zncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gate_a(records):
    scores = []
    for rec in records[:10]:
        ref = _load(rec, 'ref_path').astype(np.float32)
        search = _load(rec, 'search_path').astype(np.float32)
        w = int(round(1000.0 / rec['scale']))
        # sub-pixel-centred crop at the ground-truth location, then upscale
        crop = cv2.getRectSubPix(search, (w, w),
                                 (rec['true_x'], rec['true_y']))
        up = cv2.resize(crop, (1000, 1000), interpolation=cv2.INTER_CUBIC)
        # This gate verifies ground-truth POSITION.  The known synthetic
        # per-image rotation jitter is compensated first — without this, the
        # ZNCC measures rotation mismatch instead of coordinate correctness.
        rel = rec['search_rotation_deg'] - rec['ref_rotation_deg']
        Mr = cv2.getRotationMatrix2D((500.0, 500.0), rel, 1.0)
        ref_al = cv2.warpAffine(ref, Mr, (1000, 1000))
        c = (slice(150, 850), slice(150, 850))   # avoid warp border effects
        score = zncc(ref_al[c], up[c])
        scores.append(score)
        assert score > 0.5, f"{rec['id']}: round-trip ZNCC {score:.3f} <= 0.5"
    print(f"  (a) round-trip ZNCC on 10 pairs: min={min(scores):.3f} "
          f"mean={np.mean(scores):.3f} (all > 0.5)")


def gate_b(records):
    assert len(records) >= 40, f"only {len(records)} records"
    for rec in records:
        for k in REQUIRED_KEYS:
            assert k in rec, f"{rec.get('id', '?')}: missing key {k}"
        assert MARGIN <= rec['true_x'] <= 1000 - MARGIN, rec['id']
        assert MARGIN <= rec['true_y'] <= 1000 - MARGIN, rec['id']
        assert os.path.exists(os.path.join(OUT, rec['ref_path'])), rec['id']
        assert os.path.exists(os.path.join(OUT, rec['search_path'])), rec['id']
    print(f"  (b) {len(records)} records: all keys present, coords within "
          f"{MARGIN}px margin")


def gate_c(records):
    worst = 0.0
    for rec in records:
        with open(os.path.join(OUT, rec['ref_path']), 'rb') as f:
            ref_bytes = f.read()
        with open(os.path.join(OUT, rec['search_path']), 'rb') as f:
            search_bytes = f.read()
        assert ref_bytes != search_bytes, f"{rec['id']}: identical PNGs"
        ref = _load(rec, 'ref_path').astype(np.float32)
        search = _load(rec, 'search_path').astype(np.float32)
        # high-pass residuals are noise-dominated; reused noise arrays would
        # correlate strongly here
        hr = ref - cv2.GaussianBlur(ref, (0, 0), 2.0)
        hs = search - cv2.GaussianBlur(search, (0, 0), 2.0)
        r = abs(zncc(hr, hs))
        worst = max(worst, r)
        assert r < 0.05, f"{rec['id']}: noise correlation {r:.3f}"
    print(f"  (c) ref/search bytes differ, noise fields uncorrelated "
          f"(worst |r|={worst:.4f})")


def gate_d(stats):
    assert stats['n'] > 0 and stats['total_seconds'] > 0
    print(f"  (d) generation time: {stats['total_seconds']:.1f}s for "
          f"{stats['n']} pairs ({stats['seconds_per_pair']:.1f}s/pair)")


if __name__ == '__main__':
    print("Verify Gate 3 (generate_dataset.py)")
    recs, st = ensure_dataset()
    gate_a(recs)
    gate_b(recs)
    gate_c(recs)
    gate_d(st)
    print("PASS")
