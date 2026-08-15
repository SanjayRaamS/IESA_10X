"""Verify Gate 1 for core/layout.py.

  a) FFT-measured dominant pitch matches meta within 3% (20 DRAM + 20 FinFET).
  b) aperiodic_level=0 has lower notch-filtered residual energy than level=1.
  c) 2x2 contact sheet saved to data/preview_layouts.png.

Run: python tests/test_layout.py   (prints PASS on success)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from core.layout import render_dram, render_finfet

SHAPE = (1024, 1024)
PAD = 4           # FFT zero-padding factor (sub-bin peak location)
MIN_CYCLES = 3.0  # ignore periods longer than n/3 (DC / boundary leakage)


def dominant_pitch(img, axis):
    """Dominant non-DC spatial period (px) along axis (0=y, 1=x).

    FFT magnitudes are summed incoherently across the orthogonal axis so a
    phase seam cannot cancel the fundamental, then the peak is located with
    parabolic interpolation on the zero-padded, Hann-windowed spectrum."""
    n = img.shape[axis]
    win = np.hanning(n)
    sig = img - img.mean()
    sig = sig * (win[:, None] if axis == 0 else win[None, :])
    spec = np.abs(np.fft.rfft(sig, n=PAD * n, axis=axis)).sum(axis=1 - axis)
    k0 = int(np.ceil(MIN_CYCLES * PAD))
    k = k0 + int(np.argmax(spec[k0:]))
    y1, y2, y3 = spec[k - 1], spec[k], spec[k + 1]
    denom = y1 - 2.0 * y2 + y3
    delta = 0.5 * (y1 - y3) / denom if denom != 0.0 else 0.0
    return PAD * n / (k + delta)


def notch_residual_energy(img, pitch_x, pitch_y, hw=3.0, lowcut=4.0):
    """Mean-square residual after notching every lattice harmonic (m/pitch_x,
    n/pitch_y) and a small low-frequency box out of the 2-D spectrum."""
    H, W = img.shape
    F = np.fft.fft2(img)
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W

    def comb(f, step):
        return np.abs(f - np.round(f / step) * step) <= hw

    lattice = comb(ky, H / pitch_y)[:, None] & comb(kx, W / pitch_x)[None, :]
    low = (np.abs(ky) <= lowcut)[:, None] & (np.abs(kx) <= lowcut)[None, :]
    F[lattice | low] = 0.0
    res = np.fft.ifft2(F).real
    return float(np.mean(res * res))


def gate_a():
    worst = 0.0
    for style, fn, base in (('dram', render_dram, 1000),
                            ('finfet', render_finfet, 5000)):
        for i in range(20):
            rng = np.random.default_rng(base + i)
            level = float(rng.uniform(0.0, 1.0))
            img, meta = fn(SHAPE, rng, {'aperiodic_level': level})
            for axis, key in ((1, 'pitch_x'), (0, 'pitch_y')):
                p = dominant_pitch(img, axis)
                err = abs(p - meta[key]) / meta[key]
                worst = max(worst, err)
                assert err < 0.03, (f"{style} #{i} {key}: meta={meta[key]:.2f} "
                                    f"measured={p:.2f} err={100*err:.1f}%")
    print(f"  (a) FFT pitch matches meta within 3% on 40 layouts "
          f"(worst {100*worst:.2f}%)")


def gate_b():
    for style, fn, base in (('dram', render_dram, 300),
                            ('finfet', render_finfet, 700)):
        e0s, e1s = [], []
        for i in range(8):
            img0, m0 = fn(SHAPE, np.random.default_rng(base + i),
                          {'aperiodic_level': 0.0})
            img1, m1 = fn(SHAPE, np.random.default_rng(base + i),
                          {'aperiodic_level': 1.0})
            e0s.append(notch_residual_energy(img0, m0['pitch_x'], m0['pitch_y']))
            e1s.append(notch_residual_energy(img1, m1['pitch_x'], m1['pitch_y']))
        e0, e1 = float(np.mean(e0s)), float(np.mean(e1s))
        assert e1 > 2.0 * e0, (style, e0, e1)
        assert all(b > a for a, b in zip(e0s, e1s)), (style, e0s, e1s)
        print(f"  (b) {style}: residual energy level=0 {e0:.3e} < "
              f"level=1 {e1:.3e} (x{e1 / e0:.1f})")


def gate_c():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    specs = [("DRAM  aperiodic=0", render_dram, 0.0, 42),
             ("DRAM  aperiodic=1", render_dram, 1.0, 42),
             ("FinFET  aperiodic=0", render_finfet, 0.0, 43),
             ("FinFET  aperiodic=1", render_finfet, 1.0, 43)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (title, fn, lvl, seed) in zip(axes.ravel(), specs):
        img, _ = fn((512, 512), np.random.default_rng(seed),
                    {'aperiodic_level': lvl})
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    out = os.path.join(ROOT, "data", "preview_layouts.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    assert os.path.getsize(out) > 0
    print(f"  (c) contact sheet -> {out}")


if __name__ == "__main__":
    print("Verify Gate 1 (core/layout.py)")
    gate_a()
    gate_b()
    gate_c()
    print("PASS")
