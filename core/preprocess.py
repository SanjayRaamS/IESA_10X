"""Preprocessing for Drift-Sense (Phase 4).

prep() converts a raw uint8 SEM capture into a float32 field suitable for
correlation: variance-stabilised (SEM noise is Poisson-dominant and therefore
signal-dependent, while ZNCC assumes homoscedastic noise), illumination-
flattened, and locally contrast-normalised.

gradient_orientation_features() provides edge-polarity-invariant orientation
channels for the dual-channel correlator (Phase 5+).

Citation keys resolve in REFERENCES.md.
"""

import cv2
import numpy as np

_LCN_WINDOW = 31   # local contrast normalisation window (px)
_FLAT_SIGMA = 25.0  # illumination-flattening Gaussian sigma (px)
_EPS = 1e-2


def _wide_blur(img, sigma, factor=4):
    """Gaussian blur for large sigma, done on a decimated copy.

    The illumination/charging background is by definition smooth, so it is
    band-limited far below the full sampling rate; blurring at 1/4 resolution
    is visually identical and ~5x cheaper than a 150-tap kernel at full size."""
    if sigma < 8.0 or min(img.shape) < 8 * factor:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape
    small = cv2.resize(img, (max(w // factor, 1), max(h // factor, 1)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def anscombe(img_uint8):
    """Anscombe VST 2*sqrt(x + 3/8): Poisson noise -> ~unit-variance Gaussian
    [Anscombe1948; Makitalo2011].

    Pointwise, so it is scale-free and must be applied to the RAW counts,
    before any resampling."""
    return 2.0 * np.sqrt(np.asarray(img_uint8, dtype=np.float32) + 0.375)


def normalise(img, flat_sigma=_FLAT_SIGMA, lcn_window=_LCN_WINDOW):
    """Illumination flattening + local contrast normalisation.

    Both steps are SPATIAL, so their windows define a physical scale.  When
    comparing a reference against a search image at a different magnification,
    normalise each AFTER bringing them to common sampling — normalising the
    reference at its own magnification and then demagnifying it by ~10x
    shrinks these windows by the same factor and the two images end up
    normalised at scales that differ by an order of magnitude.
    """
    img = np.asarray(img, dtype=np.float32)
    flat = img - _wide_blur(img, flat_sigma)
    k = (int(lcn_window) | 1,) * 2
    m = cv2.boxFilter(flat, -1, k)
    v = cv2.boxFilter(flat * flat, -1, k) - m * m
    return flat / np.sqrt(np.clip(v, 0.0, None) + _EPS)


def prep(img_uint8):
    """uint8 SEM image -> float32 normalised field (VST, flatten, LCN)."""
    return normalise(anscombe(img_uint8))


def gradient_orientation_features(img):
    """(|grad|*cos(2*theta), |grad|*sin(2*theta)) of a float image.

    The DOUBLED angle makes the features invariant to edge polarity: an SEM
    bright edge band has opposite gradient signs on the two sides of the same
    physical feature, and single-angle orientation would cancel them.
    Identities: |g|cos2t = (gx^2 - gy^2)/|g|,  |g|sin2t = 2*gx*gy/|g|.
    """
    g = np.asarray(img, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    safe = mag + np.float32(1e-8)
    return (gx * gx - gy * gy) / safe, (2.0 * gx * gy) / safe


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (256, 256))).astype(np.uint8)
    out = prep(img)
    assert out.dtype == np.float32 and out.shape == img.shape

    # polarity invariance: inverting the image flips every gradient sign but
    # must leave the doubled-angle features unchanged
    fx1, fy1 = gradient_orientation_features(img.astype(np.float32))
    fx2, fy2 = gradient_orientation_features(255.0 - img.astype(np.float32))
    assert np.allclose(fx1, fx2, atol=1e-3) and np.allclose(fy1, fy2, atol=1e-3)
    print(f"prep: out std={out.std():.2f}  orientation features "
          f"polarity-invariant OK")
    print("preprocess self-check OK")
