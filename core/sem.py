"""SEM forward imaging model for Drift-Sense (Phase 2).

image_sem turns an ideal binary layout (core/layout.py) into a realistic SEM
capture.  The stage order is physical and must not be reordered: edge
brightening happens on the specimen surface, geometry and PSF in the column,
charging on the specimen, shot noise in the electron dose, readout noise in
the detector chain, quantisation in the frame grabber.

INDEPENDENT NOISE IS MANDATORY: the reference and search images are separate
physical captures — always call image_sem with two DIFFERENT rng objects,
seeded differently.  Noise arrays are drawn fresh inside every call and are
never reused.

Citation keys ([Reimer1998] etc.) resolve in REFERENCES.md.
"""

import cv2
import numpy as np

_DEFAULTS = {
    'edge_gain': 0.35,    # strength of SE edge brightening
    'rotation_deg': 0.0,  # stage rotation error
    'scale': 1.0,         # magnification factor (0.1 = 10x lower mag)
    'psf_sigma': 1.2,     # beam PSF sigma in output px
    'shading': 0.08,      # charging/shading field amplitude
    'dose': 300.0,        # electrons per pixel at signal level 1.0
    'readout': 0.02,      # detector readout noise sigma (normalised units)
}


def warp_matrix(shape, rotation_deg, scale):
    """Affine matrix + output size used by image_sem stage 2.  Exposed so
    dataset generation can map ground-truth coordinates through the exact
    transform applied to the image."""
    H, W = shape
    out_w = max(int(round(W * float(scale))), 1)
    out_h = max(int(round(H * float(scale))), 1)
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(rotation_deg),
                                float(scale))
    M[0, 2] += out_w / 2.0 - W / 2.0
    M[1, 2] += out_h / 2.0 - H / 2.0
    return M, (out_w, out_h)


def image_sem(ideal, rng, params=None, return_stages=False):
    """Forward-image an ideal layout into a uint8 SEM capture.

    ideal: float array in [0, 1].  rng: np.random.Generator (unique per
    capture).  Returns uint8 image; with return_stages=True also returns a
    dict of the intermediate float stages (used by tests and previews).
    """
    p = {**_DEFAULTS, **(params or {})}
    img = np.asarray(ideal, dtype=np.float32)
    stages = {'ideal': img.copy()}

    # 1) Edge brightening: SE yield rises with local surface tilt, so sidewalls
    #    emit more than flat tops; added BEFORE blur so the bright band is
    #    convolved with everything else [Reimer1998; Goldstein2018]
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    gmax = float(grad.max())
    if gmax > 0.0:
        grad /= gmax
    img = img + np.float32(p['edge_gain']) * grad
    stages['edge'] = img.copy()

    # 2) Geometric degradation: stage rotation + magnification change
    #    (INTER_AREA when downscaling, INTER_CUBIC otherwise)
    scale = float(p['scale'])
    M, (out_w, out_h) = warp_matrix(img.shape, p['rotation_deg'], scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img = cv2.warpAffine(img, M, (out_w, out_h), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    stages['warped'] = img.copy()

    # 3) PSF blur: beam spot + interaction volume, slightly anisotropic from
    #    raster distortion (sigma_y = sigma_x * U(0.9, 1.1)) [Postek2013]
    sx = float(p['psf_sigma'])
    sy = sx * float(rng.uniform(0.9, 1.1))
    if sx > 0.0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
    stages['blurred'] = img.copy()

    # 4) Shading/charging: insulator charging gives a slowly-varying background,
    #    modelled as a wide random Gaussian blob field [Reimer1998]
    amp = float(p['shading'])
    if amp != 0.0:
        h, w = img.shape
        cx, cy = rng.uniform(0.0, w), rng.uniform(0.0, h)
        sig = rng.uniform(0.4, 0.8) * max(h, w)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
        img = img + np.float32(sign * amp) * blob.astype(np.float32)
    stages['shaded'] = img.copy()

    # 5) Shot noise: Poisson electron counting — variance proportional to
    #    signal, the dominant noise term (why Anscombe matters downstream)
    #    [Foi2008]
    dose = float(p['dose'])
    img = rng.poisson(np.clip(img, 0.0, None) * dose).astype(np.float32) / dose
    stages['shot'] = img.copy()

    # 6) Readout noise: additive Gaussian from the detector/amplifier chain
    #    [Foi2008]
    img = img + rng.normal(0.0, float(p['readout']), img.shape).astype(np.float32)
    stages['readout'] = img.copy()

    # 7) Clip and quantise to 8-bit (frame-grabber ADC)
    out = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    stages['final'] = out
    return (out, stages) if return_stages else out


if __name__ == "__main__":
    from layout import render_dram

    ideal, _ = render_dram((256, 256), np.random.default_rng(0),
                           {'aperiodic_level': 0.5})
    a = image_sem(ideal, np.random.default_rng(1))
    b = image_sem(ideal, np.random.default_rng(2))
    assert a.dtype == np.uint8 and a.shape == ideal.shape
    assert np.any(a != b), "different rngs must give different noise"
    small = image_sem(ideal, np.random.default_rng(3), {'scale': 0.1})
    assert small.shape == (26, 26)
    flat = image_sem(np.full((64, 64), 0.5, dtype=np.float32),
                     np.random.default_rng(4), {'shading': 0.0, 'edge_gain': 0.0})
    print(f"sem: range=({a.min()},{a.max()}) mean={a.mean():.1f} "
          f"flat-patch std={flat.std():.2f} DN  10x-downscale shape={small.shape}")
    print("sem self-check OK")
