"""Lattice geometry from FFT for Drift-Sense (Phase 5).

This replaces brute-force multi-scale/multi-rotation template search with two
FFTs: the periodic component of a layout carries ZERO position information but
PERFECT scale/rotation information, so the geometry is solved analytically from
the reciprocal lattice.

Convention.  A maps reference coords to search coords: x_search = A @ x_ref.
A real-space lattice basis L (columns = primitive vectors) has reciprocal basis
B = L^-T, so L_search = A L_ref implies

    B_search = A^-T B_ref   =>   A = (B_search B_ref^-1)^-T

A reciprocal basis is only defined up to a unimodular integer matrix M acting
on the right (B' = B M — a different choice of which two lattice vectors were
picked), which is what estimate_transform resolves using the scale prior.

Citation keys resolve in REFERENCES.md.
"""

import cv2
import numpy as np

_DC_RADIUS = 3.5     # bins blanked around DC (must stay below the lowest
                     # fundamental: a 180 px gate pitch is bin ~5.6 at N=1000)
_MIN_SIN = 0.25      # reject near-parallel basis pairs
_PEAK_SEP = 4        # minimum separation between kept peaks (bins)
_REL_PROM = 0.08     # a "clean" peak is >=8% as prominent as the strongest;
                     # measured sweep: admitting weak HARMONICS materially
                     # improves the angular fit (they sit at large radius)
_SNR = 4.0           # ...and >=4 robust sigma above the spectral background
_SCALE_WINDOW = 1.7  # admissible scale range about the prior (excludes 2x)
_PRIOR_SIGMA = 0.15  # log-normal width of the scale prior used in ranking
_NEG = -1e9


def _spectrum(img):
    """Hann-windowed log magnitude spectrum, DC-centred.

    The window kills spectral leakage from the image border, which otherwise
    dominates as a cross artefact through DC and buries the lattice peaks."""
    a = np.asarray(img, dtype=np.float32)
    H, W = a.shape
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win)))
    return np.log1p(F).astype(np.float32)


def _prominence(S, dc_radius=_DC_RADIUS):
    """Local prominence: the log spectrum minus a smooth background, so the
    broad DC skirt cannot outrank genuine lattice peaks."""
    P = S - cv2.blur(S, (15, 15))
    H, W = S.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    P[np.hypot(xx - cx, yy - cy) <= dc_radius] = _NEG
    return P


def _refine(P, y, x):
    """2D parabolic fit on the 3x3 neighbourhood -> sub-bin offset and
    curvature.  Bin quantisation is the dominant error source in this stage:
    at N=1000 one bin is ~0.06 deg of lattice orientation."""
    def axis(m, c, p):
        den = m - 2.0 * c + p
        d = 0.5 * (m - p) / den if den != 0.0 else 0.0
        return float(np.clip(d, -0.5, 0.5)), -float(den)
    dx, curv_x = axis(P[y, x - 1], P[y, x], P[y, x + 1])
    dy, curv_y = axis(P[y - 1, x], P[y, x], P[y + 1, x])
    return dx, dy, 0.5 * (curv_x + curv_y)


def reciprocal_basis(img_float, n_peaks=6):
    """Reciprocal lattice basis of an image.

    Returns (B, peak_info): B is 2x2 with COLUMNS the two shortest linearly
    independent reciprocal vectors in cycles/pixel, or None if fewer than two
    clean peaks exist (weakly periodic crop).  peak_info carries the retained
    half-plane peaks and sharpness diagnostics.
    """
    S = _spectrum(img_float)
    P = _prominence(S)
    H, W = S.shape
    cy, cx = H // 2, W // 2

    # strongest local maxima (dilation-equality), strongest first
    mx = cv2.dilate(P, np.ones((5, 5), np.float32))
    ys, xs = np.nonzero((P >= mx) & (P > _NEG / 2))
    keep = (ys > 0) & (ys < H - 1) & (xs > 0) & (xs < W - 1)
    ys, xs = ys[keep], xs[keep]
    order = np.argsort(P[ys, xs])[::-1]

    peaks = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        ky, kx = y - cy, x - cx
        # conjugate symmetry: a peak and its mirror are the same lattice
        # vector, so keep only the upper half-plane
        if not (ky > 0 or (ky == 0 and kx > 0)):
            continue
        if any(abs(ky - p['bin'][0]) < _PEAK_SEP and
               abs(kx - p['bin'][1]) < _PEAK_SEP for p in peaks):
            continue
        dx, dy, curv = _refine(P, y, x)
        peaks.append({'bin': (ky, kx),
                      'freq': ((kx + dx) / W, (ky + dy) / H),
                      'prominence': float(P[y, x]),
                      'sharpness': curv})
        if len(peaks) >= n_peaks:
            break

    # Keep only CLEAN peaks before applying the shortest-vector rule: window
    # sidelobes and noise ripples are weak but can be shorter than a genuine
    # fundamental, and "shortest" would otherwise select them.
    valid = P[P > _NEG / 2]
    sigma = 1.4826 * float(np.median(np.abs(valid - np.median(valid)))) + 1e-9
    if peaks:
        floor = max(_REL_PROM * peaks[0]['prominence'], _SNR * sigma)
        peaks = [p for p in peaks if p['prominence'] >= floor]

    info = {'peaks': peaks, 'n_peaks': len(peaks), 'noise_sigma': sigma,
            'sharpness': float(np.mean([p['sharpness'] for p in peaks]))
            if peaks else 0.0}

    # two SHORTEST linearly independent vectors (harmonics of one direction
    # are parallel and get rejected by the |sin| test)
    order_len = sorted(peaks, key=lambda p: float(np.hypot(*p['freq'])))
    for i, p in enumerate(order_len):
        v1 = np.array(p['freq'], dtype=float)
        for q in order_len[i + 1:]:
            v2 = np.array(q['freq'], dtype=float)
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if cross > _MIN_SIN * np.linalg.norm(v1) * np.linalg.norm(v2):
                B = np.column_stack([v1, v2])
                info['basis_peaks'] = [p, q]
                return B, info
    return None, info


def _unimodular():
    """All 2x2 integer matrices with entries in [-2, 2] and |det| = 1."""
    r = range(-2, 3)
    return [np.array([[a, b], [c, d]], dtype=float)
            for a in r for b in r for c in r for d in r
            if abs(a * d - b * c) == 1]


_UNIMODULAR = _unimodular()


def _decompose(A):
    """Polar decomposition of a 2x2: (scale, rotation_deg, anisotropy).

    Rotation is reported in the same convention as cv2.getRotationMatrix2D,
    i.e. A = s * [[cos, sin], [-sin, cos]] for a positive angle."""
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt
    rot = np.degrees(np.arctan2(-R[1, 0], R[0, 0]))
    return float(np.sqrt(sv[0] * sv[1])), float(rot), float(sv[0] / sv[1])


def _logpolar_transform(spec, n_ang=720, n_rad=512):
    h, w = spec.shape
    maxr = 0.5 * min(h, w)
    lp = cv2.warpPolar(spec, (n_rad, n_ang), (w / 2.0, h / 2.0), maxr,
                       cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG)
    return lp.astype(np.float32), n_rad / np.log(maxr)


def _logpolar_fallback(ref, search, scale_prior):
    """Log-polar phase correlation between the two magnitude spectra
    [Reddy1996].  Rotation becomes a shift in angle, scale a shift in
    log-radius, so one phase correlation recovers both."""
    Sr, Ss = _spectrum(ref), _spectrum(search)
    for S in (Sr, Ss):
        h, w = S.shape
        yy, xx = np.mgrid[0:h, 0:w]
        S[np.hypot(xx - w // 2, yy - h // 2) <= _DC_RADIUS] = 0.0
    lp_r, klog = _logpolar_transform(Sr)
    lp_s, _ = _logpolar_transform(Ss)
    win = cv2.createHanningWindow((lp_r.shape[1], lp_r.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(lp_r.astype(np.float64),
                                        lp_s.astype(np.float64),
                                        win.astype(np.float64))
    n_ang = lp_r.shape[0]
    # spectra have 180 deg symmetry: choose the branch nearest zero rotation
    rot = dy * 360.0 / n_ang
    rot = (rot + 90.0) % 180.0 - 90.0
    # search spectrum expands by 1/scale, so a positive log-radius shift means
    # a real-space demagnification
    scale = float(np.exp(-dx / klog))
    A = scale * np.array([[np.cos(np.radians(rot)), np.sin(np.radians(rot))],
                          [-np.sin(np.radians(rot)), np.cos(np.radians(rot))]])
    diag = {'method': 'logpolar', 'scale': scale, 'rotation_deg': float(rot),
            'anisotropy': 1.0, 'sharpness': float(resp), 'response': float(resp),
            'n_peaks_ref': 0, 'n_peaks_search': 0, 'residual': float('nan')}
    return A, diag


def _as_complex(peaks):
    return np.array([complex(p['freq'][0], p['freq'][1]) for p in peaks])


def _c_to_A(c):
    """Complex ratio c (search reciprocal / ref reciprocal) -> real-space A.

    Reciprocal vectors map as f_search = A^-T f_ref.  Under a similarity
    A = s*R this is multiplication by the complex number c = e^-i(theta)/s,
    so scale = 1/|c| and rotation = -arg(c)."""
    s = 1.0 / abs(c)
    th = -np.angle(c)
    return s * np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])


def _match(c, zr, wr, zs, ws, n):
    """Match ref reciprocal vectors to search ones under ratio c.

    Returns (weighted score, matched ref vectors, matched search vectors).
    Both signs of each search peak are tried: the half-plane convention can
    flip a vector relative to its partner in the other image."""
    score, mr, ms = 0.0, [], []
    tol_abs, tol_rel = 3.0 / n, 0.015
    for z, w in zip(zr, wr):
        pred = c * z
        tol = tol_abs + tol_rel * abs(pred)
        d = np.minimum(np.abs(zs - pred), np.abs(-zs - pred))
        i = int(np.argmin(d))
        if d[i] < tol:
            score += min(w, ws[i])
            mr.append(z)
            ms.append(zs[i] if abs(zs[i] - pred) <= abs(-zs[i] - pred)
                      else -zs[i])
    return score, np.array(mr), np.array(ms)


def _lsq_ratio(mr, ms, w):
    """Weighted complex least squares for c in ms = c * mr.

    Solving in the complex plane enforces the similarity constraint (equal
    scale on both axes, no shear) exactly, and high-order harmonics — which
    sit at large radius — dominate the fit, so the angular precision is far
    better than any single fundamental can give."""
    num = np.sum(w * ms * np.conj(mr))
    den = np.sum(w * np.abs(mr) ** 2)
    return num / den if abs(den) > 1e-20 else None


def estimate_transform(ref_float, search_float, scale_prior=0.1):
    """Recover the 2x2 linear map from reference coords to search coords.

    Returns (A, diag).  diag carries method, scale, rotation_deg, anisotropy,
    peak sharpness and the fit residual.

    The spec's basis + unimodular route runs first and provides one hypothesis.
    It is not trusted alone: on real data the shortest reciprocal vector is
    often a sub-array-break SUPERLATTICE peak that only one of the two images
    resolves, and at search magnification a fine direction (FinFET fins at
    ~2.5 px/period) can be erased entirely by the PSF, so no common 2D basis
    exists.  Every (ref peak, search peak) assignment is therefore also taken
    as a hypothesis; each is scored by how many OTHER peaks it explains, and
    the winner is refined by similarity least squares over all its matches.
    A single shared direction plus its harmonics is enough, which is what
    keeps the FinFET pairs on the analytic path instead of the fallback.
    """
    B_ref, info_r = reciprocal_basis(ref_float, n_peaks=24)
    B_search, info_s = reciprocal_basis(search_float, n_peaks=24)
    pr, ps = info_r['peaks'], info_s['peaks']
    n = np.shape(ref_float)[0]

    if not pr or not ps:
        A, diag = _logpolar_fallback(ref_float, search_float, scale_prior)
        diag.update(reason='no_peaks', n_peaks_ref=len(pr), n_peaks_search=len(ps))
        return A, diag

    zr, zs = _as_complex(pr), _as_complex(ps)
    wr = np.array([p['prominence'] for p in pr])
    ws = np.array([p['prominence'] for p in ps])

    # A factor-2 error is the failure mode here: matching a ref fundamental to
    # a search HARMONIC is internally consistent (2f -> 4f also lands on a
    # peak) and scores well.  The admissible scale window must therefore
    # exclude 2x and 1/2x while still allowing real scale variation about the
    # prior — the prior is a window, never a hardcoded ratio.
    def plausible(c):
        s = 1.0 / abs(c)
        return (scale_prior / _SCALE_WINDOW < s < scale_prior * _SCALE_WINDOW
                and abs(np.degrees(-np.angle(c))) < 15.0)

    def rank(entry):
        """Match score weighted by a soft log-normal scale prior.

        A hard window alone leaves hypotheses near its edge competitive; this
        makes the prior do real work — a 2x harmonic mismatch is suppressed by
        ~1e-5 while genuine scale variation (a few tens of percent) is barely
        touched, so the window can stay wide."""
        s = 1.0 / abs(entry[1])
        return entry[0] * np.exp(-0.5 * (np.log(s / scale_prior) / _PRIOR_SIGMA) ** 2)

    cands, uni_c = [], None
    if B_ref is not None and B_search is not None:
        # spec route: resolve the unimodular ambiguity with the scale prior
        Bri = np.linalg.inv(B_ref)
        target = scale_prior * np.eye(2)
        best = None
        for M in _UNIMODULAR:
            C = B_search @ M @ Bri
            if abs(np.linalg.det(C)) < 1e-12:
                continue
            A = np.linalg.inv(C).T
            if np.linalg.det(A) <= 0:      # reflections are not physical
                continue
            cost = float(np.linalg.norm(A - target, ord='fro'))
            if best is None or cost < best[0]:
                best = (cost, A)
        if best is not None:
            sc, rt, _ = _decompose(best[1])
            uni_c = np.exp(-1j * np.radians(rt)) / sc
            if plausible(uni_c):
                cands.append(uni_c)

    # every peak-to-peak assignment is a similarity hypothesis
    for z in zr:
        if abs(z) < 1e-9:
            continue
        for zz in zs:
            for sgn in (1.0, -1.0):
                c = (sgn * zz) / z
                if plausible(c):
                    cands.append(c)

    scored = []
    for c in cands:
        sc, mr, ms = _match(c, zr, wr, zs, ws, n)
        if len(mr) >= 1:
            scored.append((sc, c, mr, ms))
    if not scored:
        A, diag = _logpolar_fallback(ref_float, search_float, scale_prior)
        diag.update(reason='no_consistent_hypothesis',
                    n_peaks_ref=len(pr), n_peaks_search=len(ps))
        return A, diag

    score, c, mr, ms = max(scored, key=rank)
    for _ in range(2):                      # refine, re-match, refine
        c_new = _lsq_ratio(mr, ms, np.abs(mr))
        if c_new is None or not plausible(c_new):
            break
        c = c_new
        s2, mr2, ms2 = _match(c, zr, wr, zs, ws, n)
        if len(mr2) >= 1:
            score, mr, ms = s2, mr2, ms2

    A = _c_to_A(c)
    scale, rot, aniso = _decompose(A)
    resid = float(np.sqrt(np.mean(np.abs(ms - c * mr) ** 2))) if len(mr) else 0.0
    diag = {'method': 'lattice' if uni_c is not None and abs(c - uni_c) < 1e-12
            else 'lattice_matched',
            'scale': scale, 'rotation_deg': rot, 'anisotropy': aniso,
            'sharpness': 0.5 * (info_r['sharpness'] + info_s['sharpness']),
            'n_peaks_ref': len(pr), 'n_peaks_search': len(ps),
            'n_matched': int(len(mr)), 'match_score': float(score),
            'residual': resid * n}
    return A, diag


if __name__ == "__main__":
    # synthetic ground truth pins the sign conventions without guessing
    rng = np.random.default_rng(0)
    N = 1000
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    base = (np.sin(2 * np.pi * xx / 40.0) + np.sin(2 * np.pi * yy / 55.0)
            ).astype(np.float32)
    B, info = reciprocal_basis(base)
    # columns are sorted by length, not by axis: recover both pitches as 1/|b|
    pitches = sorted(1.0 / np.linalg.norm(B[:, i]) for i in range(2))
    px, py = pitches
    assert abs(px - 40.0) < 0.5 and abs(py - 55.0) < 0.5, pitches

    true_scale, true_rot = 0.1, 2.5
    M = cv2.getRotationMatrix2D((N / 2.0, N / 2.0), true_rot, true_scale)
    M[0, 2] += 0.5 * (N * true_scale) - N / 2.0
    M[1, 2] += 0.5 * (N * true_scale) - N / 2.0
    small = cv2.warpAffine(base, M, (int(N * true_scale), int(N * true_scale)),
                           flags=cv2.INTER_AREA)
    A, diag = estimate_transform(base, small, scale_prior=0.1)
    print(f"lattice: pitches=({px:.2f},{py:.2f}) "
          f"method={diag['method']} scale={diag['scale']:.5f} "
          f"(true {true_scale}) rot={diag['rotation_deg']:+.3f} "
          f"(true {true_rot:+.1f}) aniso={diag['anisotropy']:.3f}")
    assert abs(diag['scale'] - true_scale) / true_scale < 0.02
    assert abs(diag['rotation_deg'] - true_rot) < 0.3
    print("lattice self-check OK")
