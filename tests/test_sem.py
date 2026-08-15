"""Verify Gate 2 for core/sem.py.

  a) Poisson check: variance vs mean over flat patches is linear with
     positive slope (R^2 > 0.95) — noise is signal-dependent.
  b) Independence check: two captures of the same layout with different
     seeds have noise fields with Pearson |r| < 0.05.
  c) Edge check: a 3px band around ideal-layout edges is brighter than
     feature interiors.
  d) Stage-by-stage preview saved to data/preview_sem.png.

Run: python tests/test_sem.py   (prints PASS on success)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.layout import render_dram
from core.sem import image_sem


def gate_a():
    levels = np.linspace(0.05, 0.85, 10)
    p = {'edge_gain': 0.0, 'rotation_deg': 0.0, 'scale': 1.0, 'psf_sigma': 1.0,
         'shading': 0.0, 'dose': 200.0, 'readout': 0.01}
    means, variances = [], []
    for i, lv in enumerate(levels):
        ideal = np.full((160, 160), lv, dtype=np.float32)
        out = image_sem(ideal, np.random.default_rng(10 + i), p)
        out = out.astype(np.float64) / 255.0
        means.append(out.mean())
        variances.append(out.var())
    means, variances = np.asarray(means), np.asarray(variances)
    slope, icept = np.polyfit(means, variances, 1)
    pred = slope * means + icept
    r2 = 1.0 - np.sum((variances - pred) ** 2) / np.sum(
        (variances - variances.mean()) ** 2)
    assert slope > 0.0, f"variance-vs-mean slope {slope:.3e} not positive"
    assert r2 > 0.95, f"linear fit R^2={r2:.3f} <= 0.95"
    print(f"  (a) shot noise signal-dependent: var = {slope:.2e}*mean + "
          f"{icept:.2e}, R^2={r2:.4f}")


def gate_b():
    ideal, _ = render_dram((512, 512), np.random.default_rng(1),
                           {'aperiodic_level': 0.5})
    p = {'rotation_deg': 0.0, 'scale': 1.0}
    out1, st1 = image_sem(ideal, np.random.default_rng(101), p,
                          return_stages=True)
    out2, st2 = image_sem(ideal, np.random.default_rng(202), p,
                          return_stages=True)
    # noise field = final capture minus that capture's own noise-free signal
    n1 = out1.astype(np.float64) / 255.0 - np.clip(st1['shaded'], 0.0, 1.0)
    n2 = out2.astype(np.float64) / 255.0 - np.clip(st2['shaded'], 0.0, 1.0)
    r = float(np.corrcoef(n1.ravel(), n2.ravel())[0, 1])
    assert abs(r) < 0.05, f"noise fields correlated: r={r:.4f}"
    print(f"  (b) independent captures: noise Pearson r={r:+.4f} (|r| < 0.05)")


def gate_c():
    ideal, _ = render_dram((512, 512), np.random.default_rng(7),
                           {'aperiodic_level': 0.0})
    p = {'edge_gain': 0.5, 'rotation_deg': 0.0, 'scale': 1.0, 'psf_sigma': 1.0,
         'shading': 0.0, 'dose': 400.0, 'readout': 0.01}
    out = image_sem(ideal, np.random.default_rng(8), p).astype(np.float64) / 255.0
    gx = cv2.Sobel(ideal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ideal, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    # edge locus = strong-gradient pixels (the sidewalls), dilated to a 3px band
    edge = (grad > 0.5 * grad.max()).astype(np.uint8)
    band = cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0
    interior = (ideal > 0.1) & ~band
    m_band, m_int = out[band].mean(), out[interior].mean()
    assert m_band > m_int, f"edge band {m_band:.3f} !> interior {m_int:.3f}"
    print(f"  (c) edge brightening: band mean {m_band:.3f} > interior mean "
          f"{m_int:.3f}")


def gate_d():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ideal, _ = render_dram((512, 512), np.random.default_rng(21),
                           {'aperiodic_level': 0.7})
    p = {'edge_gain': 0.35, 'rotation_deg': 4.0, 'scale': 0.85,
         'psf_sigma': 1.5, 'shading': 0.10, 'dose': 250.0, 'readout': 0.02}
    _, st = image_sem(ideal, np.random.default_rng(22), p, return_stages=True)
    panels = [('ideal', '1. ideal layout'),
              ('edge', '2. + edge brightening'),
              ('warped', '3. rotate + scale'),
              ('blurred', '4. PSF blur'),
              ('shaded', '5. + shading/charging'),
              ('shot', '6. + shot noise'),
              ('readout', '7. + readout noise'),
              ('final', '8. final uint8')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        img = st[key]
        img = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.3, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out = os.path.join(ROOT, "data", "preview_sem.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    assert os.path.getsize(out) > 0
    print(f"  (d) stage preview -> {out}")


if __name__ == "__main__":
    print("Verify Gate 2 (core/sem.py)")
    gate_a()
    gate_b()
    gate_c()
    gate_d()
    print("PASS")
