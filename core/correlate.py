"""Dual-channel score maps for Drift-Sense (Phase 6).

The periodic component of a layout is identical in every unit cell, so a
correlation map built from it has many equally good peaks.  The aperiodic
residual (array edges, sub-array breaks, defects) is the only channel carrying
position information.  This module builds both and multiplies them: the full
channel gives a sharp peak, the residual channel acts as an envelope selecting
the structurally correct cell among the identical-looking ones.

Score maps are padded back to search-image coordinates, so a peak index maps
directly to an (x, y) centre in the search image.
"""

import cv2
import numpy as np

try:                                  # importable as a package or as a script
    from core.lattice import _prominence, _spectrum
    from core.preprocess import gradient_orientation_features
except ImportError:
    from lattice import _prominence, _spectrum
    from preprocess import gradient_orientation_features

# Residual envelope weight.  Calibrated by sweep and CONFIRMED on a fresh
# 40-pair set generated with a different seed (accuracy within 5 px):
#
#     lam   0.00   0.25   0.50   0.75   1.00
#     train 40.0%  50.0%  52.5%  52.5%  55.0%
#     val   47.5%  60.0%  60.0%  60.0%  62.5%
#
# lam=0 -- discarding the aperiodic channel -- is decisively worse on BOTH
# sets, which is the thesis measured rather than asserted.  Between 0.25 and
# 1.0 the differences are within the noise of n=40; 1.0 is best or tied-best
# on both, so it is the shipped value.
LAM = 1.0
_NOTCH_DF = 0.006    # notch radius in cycles/px, scaled to peak width
_MIN_NOTCH_BINS = 1.5
_PSF_BAND = (0.03, 0.30)   # cycles/px band used for PSF matching


def _odd(n):
    return n if n % 2 == 1 else n - 1


def _inscribed_halfsize(mask, cy, cx):
    """Largest symmetric half-size about (cy, cx) fully inside mask.

    A rotated warp leaves zero-filled corners; TM_CCOEFF_NORMED has no mask
    support, so those corners must be cropped away rather than tolerated."""
    hmax = int(min(cy, mask.shape[0] - 1 - cy, cx, mask.shape[1] - 1 - cx))

    def ok(h):
        return bool(mask[cy - h:cy + h + 1, cx - h:cx + h + 1].all())

    if not ok(0):
        return 0
    lo, hi = 0, hmax                      # coverage is monotonic in h
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def build_template(ref_float, A):
    """Warp the reference by A into search-image sampling.

    INTER_AREA for the ~10x reduction: area averaging matches how a detector
    integrates over a larger pixel footprint, and it divides the reference
    noise sigma by ~10 — free SNR.  The output has ODD dimensions with the
    reference centre on the centre pixel, so matchTemplate indices map to
    exact centres.
    """
    ref = np.asarray(ref_float, dtype=np.float32)
    H, W = ref.shape
    A = np.asarray(A, dtype=np.float64)
    corners = np.array([[0.0, 0.0], [W, 0.0], [0.0, H], [W, H]]) @ A.T
    ext = corners.max(axis=0) - corners.min(axis=0)
    out_w, out_h = int(np.ceil(ext[0])) + 2, int(np.ceil(ext[1])) + 2
    out_w, out_h = _odd(out_w), _odd(out_h)

    # place the reference centre exactly on the template centre pixel
    rc = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    tc = np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A
    M[:, 2] = tc - A @ rc

    interp = cv2.INTER_AREA if abs(np.linalg.det(A)) < 1.0 else cv2.INTER_CUBIC
    tmpl = cv2.warpAffine(ref, M, (out_w, out_h), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    cover = cv2.warpAffine(np.ones_like(ref), M, (out_w, out_h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    h = _inscribed_halfsize(cover > 0.5, (out_h - 1) // 2, (out_w - 1) // 2)
    cy, cx = (out_h - 1) // 2, (out_w - 1) // 2
    return np.ascontiguousarray(tmpl[cy - h:cy + h + 1, cx - h:cx + h + 1])


def _radial_logspectrum(img, nbins=64):
    """Mean log magnitude in radial frequency bins (cycles/px)."""
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win)))
    S = np.log(F + 1e-6)          # true log: Gaussian blur is then ADDITIVE
    yy, xx = np.mgrid[0:h, 0:w]
    f = np.hypot((xx - w // 2) / w, (yy - h // 2) / h)
    lo, hi = _PSF_BAND
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(f.ravel(), edges) - 1
    v = S.ravel()
    keep = (idx >= 0) & (idx < nbins)
    sums = np.bincount(idx[keep], weights=v[keep], minlength=nbins)
    cnts = np.bincount(idx[keep], minlength=nbins)
    out = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return out, 0.5 * (edges[:-1] + edges[1:])


def match_psf(template, search_float, max_sigma=3.0):
    """Blur the template so its spectral falloff matches the search image.

    A PSF mismatch blunts the correlation peak and silently costs accuracy.
    The template is essentially always the sharper of the two — its PSF was
    demagnified ~10x along with the pattern — so blurring it is the physical
    direction.  If the search is somehow sharper, the template is returned
    unchanged rather than sharpened (deconvolution would amplify noise).
    """
    ts, freq = _radial_logspectrum(template)
    ss, _ = _radial_logspectrum(search_float)
    good = np.isfinite(ts) & np.isfinite(ss)
    if good.sum() < 8:
        return template
    # Gaussian blur multiplies the spectrum by exp(-2 pi^2 sigma^2 f^2), which
    # is ADDITIVE in the log: log|F_blur| = log|F| - (2 pi^2 f^2) sigma^2.
    # Matching the two log profiles is therefore a one-parameter linear least
    # squares in sigma^2 -- solved in closed form instead of by grid search.
    t = ts[good] - ts[good].mean()
    s = ss[good] - ss[good].mean()
    g = 2.0 * np.pi ** 2 * freq[good] ** 2
    g = g - g.mean()
    denom = float(g @ g)
    if denom <= 1e-12:
        return template
    sigma2 = float((t - s) @ g / denom)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    if not np.isfinite(sigma) or sigma <= 1e-3:
        return template
    return cv2.GaussianBlur(template, (0, 0), min(sigma, max_sigma))


def _lattice_peak_bins(img, n_peaks=40):
    """Half-plane lattice peak bins (ky, kx) of an image, strongest first."""
    P = _prominence(_spectrum(img))
    h, w = P.shape
    cy, cx = h // 2, w // 2
    mx = cv2.dilate(P, np.ones((5, 5), np.float32))
    ys, xs = np.nonzero((P >= mx) & (P > -1e8))
    keep = (ys > 0) & (ys < h - 1) & (xs > 0) & (xs < w - 1)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return []
    order = np.argsort(P[ys, xs])[::-1]
    vals = P[ys, xs]
    floor = max(0.08 * float(vals.max()), 0.0)
    out = []
    for i in order[:n_peaks * 3]:
        if vals[i] < floor:
            break
        ky, kx = int(ys[i] - cy), int(xs[i] - cx)
        if not (ky > 0 or (ky == 0 and kx > 0)):
            continue
        out.append((ky, kx))
        if len(out) >= n_peaks:
            break
    return out


def notch_lattice(img, peak_bins=None, df=_NOTCH_DF):
    """Zero the lattice peaks (and their conjugate mirrors) out of the
    spectrum and invert: what remains is the aperiodic residual."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    if peak_bins is None:
        peak_bins = _lattice_peak_bins(img)
    F = np.fft.fftshift(np.fft.fft2(img))
    cy, cx = h // 2, w // 2
    # radius scaled to peak width: a peak occupies ~df cycles/px, which is
    # df*N bins, but never less than a bin and a half
    r = max(_MIN_NOTCH_BINS, df * min(h, w))
    ry = int(np.ceil(r)) + 1
    yy, xx = np.mgrid[-ry:ry + 1, -ry:ry + 1]
    disc = (yy * yy + xx * xx) <= r * r
    for ky, kx in peak_bins:
        for sy, sx in ((ky, kx), (-ky, -kx)):
            y0, x0 = cy + sy - ry, cx + sx - ry
            ys0, xs0 = max(y0, 0), max(x0, 0)
            ys1, xs1 = min(y0 + 2 * ry + 1, h), min(x0 + 2 * ry + 1, w)
            if ys0 >= ys1 or xs0 >= xs1:
                continue
            sub = disc[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            F[ys0:ys1, xs0:xs1][sub] = 0.0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)


def _pad_to_search(m, search_shape, tmpl_shape):
    """Pad a matchTemplate map back to search coordinates, so that
    m[y, x] is the score for a template CENTRED on search pixel (x, y)."""
    Hs, Ws = search_shape
    Ht, Wt = tmpl_shape
    top, left = (Ht - 1) // 2, (Wt - 1) // 2
    out = np.full((Hs, Ws), -np.inf, dtype=np.float32)
    out[top:top + m.shape[0], left:left + m.shape[1]] = m
    return out


def _ccoeff(search, tmpl):
    return cv2.matchTemplate(np.ascontiguousarray(search),
                             np.ascontiguousarray(tmpl), cv2.TM_CCOEFF_NORMED)


def score_maps(template, search_float, tmpl_peaks=None, search_peaks=None,
               cache=None):
    """(S_full, S_res) in search-image coordinates.

    S_full: TM_CCOEFF_NORMED on preprocessed intensity plus both
            gradient-orientation channels, averaged (FFT-based, ~10 ms here).
    S_res:  the same on the aperiodic residual, after Fourier-notching the
            lattice peaks out of BOTH template and search.
    """
    search = np.asarray(search_float, dtype=np.float32)
    tmpl = np.asarray(template, dtype=np.float32)
    shape_s, shape_t = search.shape, tmpl.shape

    acc = _ccoeff(search, tmpl)
    gsx, gsy = gradient_orientation_features(search)
    gtx, gty = gradient_orientation_features(tmpl)
    acc = acc + _ccoeff(gsx, gtx) + _ccoeff(gsy, gty)
    S_full = _pad_to_search(acc / 3.0, shape_s, shape_t)

    res_t = notch_lattice(tmpl, tmpl_peaks)
    res_s = notch_lattice(search, search_peaks)
    S_res = _pad_to_search(_ccoeff(res_s, res_t), shape_s, shape_t)
    if cache is not None:      # notching the search costs ~100 ms; let the
        cache['res_t'] = res_t  # caller reuse it instead of repeating it
        cache['res_s'] = res_s
    return S_full, S_res


def minmax_normalise(m):
    finite = np.isfinite(m)
    if not finite.any():
        return np.zeros_like(m)
    lo = float(m[finite].min())
    hi = float(m[finite].max())
    out = np.zeros_like(m, dtype=np.float32)
    out[finite] = (m[finite] - lo) / (hi - lo + 1e-12)
    return out


def combine(S_full, S_res, lam=LAM):
    """S = S_full * (1 + lam * minmax(S_res)).

    The residual map is an ENVELOPE, not an additive vote: it modulates the
    sharp periodic peaks rather than competing with them."""
    S = S_full * (1.0 + lam * minmax_normalise(S_res))
    S[~np.isfinite(S_full)] = -np.inf
    return S


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.lattice import estimate_transform
    from core.preprocess import prep

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = cv2.imread(os.path.join(root, 'data/train/pair_0000_ref.png'), 0)
    sea = cv2.imread(os.path.join(root, 'data/train/pair_0000_search.png'), 0)
    pr, ps = prep(ref), prep(sea)
    A, d = estimate_transform(pr, ps, 0.1)
    t = build_template(pr, A)
    t = match_psf(t, ps)
    S_full, S_res = score_maps(t, ps)
    S = combine(S_full, S_res)
    y, x = np.unravel_index(int(np.argmax(S)), S.shape)
    assert t.shape[0] % 2 == 1 and t.shape[1] % 2 == 1, "template must be odd"
    print(f"correlate: template={t.shape} scale={d['scale']:.4f} "
          f"argmax=({x},{y}) S_full_max={np.nanmax(S_full):.3f} "
          f"S_res_max={np.nanmax(S_res[np.isfinite(S_res)]):.3f}")
    print("correlate self-check OK")
