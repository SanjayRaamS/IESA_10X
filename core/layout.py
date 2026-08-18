"""Ideal binary layouts for Drift-Sense (Phase 1).

render_dram / render_finfet produce the IDEAL layout (no noise, no blur) at
high magnification: a float32 image in [0, 1] plus a meta dict recording the
lattice pitches and every aperiodic feature injected.

Core thesis (from the brief): the periodic lattice carries scale/rotation only;
localisation depends entirely on aperiodic structure.  params['aperiodic_level']
in [0, 1] controls how much is injected; at 0.0 the output is a pure lattice
(used to demonstrate the template-matching failure mode).

Design notes:
  - Sub-array break gaps are integer multiples of the pitch (skipped line
    positions).  A fractional gap every N lines would shift the FFT fundamental
    to pitch + gap/N; real sub-array breaks preserve the litho grid.
  - Line-edge roughness is a Phase-2 hook (_smooth_noise): currently a slight
    smooth per-row displacement of each line family, zero-mean so it never
    biases the lattice frequency.
"""

import numpy as np

_FIELD = 0.0
_DRAM_LINE = 0.55
_DRAM_CONTACT = 0.95
_FIN = 0.45
_GATE = 0.50          # fin+gate crossing renders at 0.95 (topography), brighter than either alone
_PARTICLE = 1.0


def _gaussian1d(x, sigma):
    """Periodic 1-D Gaussian smoothing (numpy-only stand-in for scipy)."""
    r = int(np.ceil(4.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([x[-r:], x, x[:r]])
    return np.convolve(xp, k, mode="same")[r:-r]


def _smooth_noise(n, corr_len, amp, rng):
    """Phase-2 hook for line-edge roughness: smooth zero-mean displacement
    field (px) sampled along the orthogonal axis."""
    if amp <= 0.0:
        return np.zeros(n)
    z = _gaussian1d(rng.standard_normal(n), corr_len)
    s = z.std()
    return z * (amp / s) if s > 1e-12 else z


def _line_centers(length, pitch, rng, pad, break_period=None, break_gap=0.0):
    """Line centre positions covering [-pad, length+pad), with an optional
    wider-than-normal gap inserted every break_period lines."""
    pos = -pad + rng.uniform(0.0, pitch)
    centers = []
    i = 0
    while pos < length + pad:
        centers.append(pos)
        i += 1
        pos += pitch
        if break_period is not None and i % break_period == 0:
            pos += break_gap
    return np.asarray(centers)


def _coverage_profile(n, centers, halfwidth):
    """Anti-aliased 1-D pixel coverage in [0, 1] of lines at `centers`."""
    prof = np.zeros(n)
    px = np.arange(n, dtype=float)
    for c in centers:
        prof += np.clip(np.minimum(px + 1.0, c + halfwidth)
                        - np.maximum(px, c - halfwidth), 0.0, 1.0)
    return np.clip(prof, 0.0, 1.0)


def _sample_shifted(profile_ext, pad, length, shifts, block=256):
    """Sample an extended 1-D profile at x - shift(row); output (len(shifts), length).

    Processed in row blocks so canvases of ~10000x10000 never materialise
    full-size float64 temporaries (dataset generation RAM budget)."""
    out = np.empty((shifts.size, length), dtype=np.float32)
    x = np.arange(length, dtype=float) + pad
    for j0 in range(0, shifts.size, block):
        j1 = min(j0 + block, shifts.size)
        coord = x[None, :] - shifts[j0:j1, None]
        i0 = np.clip(np.floor(coord).astype(np.int64), 0, profile_ext.size - 2)
        frac = np.clip(coord - i0, 0.0, 1.0)
        out[j0:j1] = profile_ext[i0] * (1.0 - frac) + profile_ext[i0 + 1] * frac
    return out


def _render_family(img_len, ortho_len, pitch, halfwidth, rng, rough_amp,
                   seam=None, brk=None):
    """Render one family of parallel lines.

    seam: (ortho_pos, shift_px) — lattice offsets by shift_px past the seam.
    brk:  (period_lines, gap_px) — extra gap every period_lines lines.
    Returns (coverage (ortho_len, img_len), centers, shifts) where shifts[j]
    is the total lattice displacement at orthogonal row j.
    """
    pad = int(np.ceil(1.6 * pitch)) + 4
    centers = _line_centers(img_len, pitch, rng, pad,
                            break_period=(brk[0] if brk else None),
                            break_gap=(brk[1] if brk else 0.0))
    prof = _coverage_profile(img_len + 2 * pad, centers + pad, halfwidth)
    shifts = _smooth_noise(ortho_len, 6.0, rough_amp, rng)
    idx = np.arange(ortho_len)
    for pos, shift in (seam or []):     # seams accumulate along the axis
        shifts = shifts + shift * (idx >= pos)
    return _sample_shifted(prof, pad, img_len, shifts), centers, shifts


def _add_disc(img, cx, cy, r, level):
    H, W = img.shape
    x0, x1 = max(int(np.floor(cx - r)) - 1, 0), min(int(np.ceil(cx + r)) + 2, W)
    y0, y1 = max(int(np.floor(cy - r)) - 1, 0), min(int(np.ceil(cy + r)) + 2, H)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cov = np.clip(r - np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) + 0.5, 0.0, 1.0)
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], cov * level)


def _paint_rect(img, x0, x1, y0, y1, value):
    H, W = img.shape
    x0, x1 = max(int(x0), 0), min(int(x1), W)
    y0, y1 = max(int(y0), 0), min(int(y1), H)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = value


def _break_positions(centers, pitch, length):
    """Mid-gap coordinates of sub-array breaks inside [0, length)."""
    d = np.diff(centers)
    mids = (0.5 * (centers[:-1] + centers[1:]))[d > 1.25 * pitch]
    return [float(m) for m in mids if 0.0 <= m < length]


_TILE = 1024.0   # the size the aperiodic feature counts below are quoted for
_MAX_SEAMS = 2   # per line family (see _seams)
_LARGE_PER_TILE = 2.5   # large-particle density (see _large_particles)


def _lin_tiles(n):
    """Canvas extent in reference tiles (>=1): 1-D features scale with it."""
    return max(1.0, float(n) / _TILE)


def _area_tiles(shape):
    """Canvas area in reference tiles (>=1): point features scale with it.

    Aperiodic features are a DENSITY, not a fixed count per image.  A dataset
    canvas is ~10000 px square, ~95 tiles, and quoting a flat 0-5 defects for
    it would leave a 1000 px reference crop with essentially no aperiodic
    content — unlocalisable in principle, which is the whole point of the
    project.  At 1024x1024 both factors are 1 and the counts are unchanged."""
    return max(1.0, shape[0] * shape[1] / (_TILE * _TILE))


def _seams(rng, prob, ortho_len, pitch):
    """Pitch phase-shift seams: the lattice offsets by a fraction of a pitch
    across a seam line.  Count scales with the canvas extent but is CAPPED:
    each seam smears the FFT fundamental, and past a couple of them the
    reciprocal lattice degrades faster than the localisation cue improves."""
    n = min(int(rng.binomial(max(1, int(round(_lin_tiles(ortho_len)))), prob)),
            _MAX_SEAMS)
    return [(int(rng.uniform(0.05, 0.95) * ortho_len),
             float(pitch * rng.uniform(0.15, 0.35))) for _ in range(n)]


def _large_particles(img, rng, level, ap, pitch, shape):
    """Large bridging particles: the navigation landmarks.

    Only features bigger than ~10x the search pixel survive the ~10x
    demagnification of the search image — a missing contact (~1 px there) or
    a small particle does not, so the point-defect classes above cannot break
    the lattice tie at low magnification.  These can, and unlike a street
    (which constrains one axis) a blob constrains BOTH.  Emitted only on
    canvases much larger than a tile, so single-tile renders are unchanged."""
    if level <= 0 or _area_tiles(shape) < 4.0:
        return
    H, W = shape
    # ~2.5 per tile at full level, so a reference-sized crop almost always
    # contains one.  A radius of 1-3 pitches is ~5-13 px in the search image:
    # comfortably resolvable there while covering only a few percent of the
    # canvas, which many larger blobs would not.
    n = int(rng.binomial(int(round(_LARGE_PER_TILE * _area_tiles(shape))),
                         0.8 * level))
    for _ in range(n):
        x, y = float(rng.uniform(0, W)), float(rng.uniform(0, H))
        r = float(rng.uniform(1.0, 3.0) * pitch)
        _add_disc(img, x, y, r, _PARTICLE)
        ap.append({'type': 'defect', 'kind': 'particle', 'size': 'large',
                   'pos': (x, y), 'radius_px': r})


def _maybe_break(rng, prob, pitch):
    if rng.random() < prob:
        # gap is an integer number of skipped line positions: keeps the FFT
        # fundamental exactly at 1/pitch (litho grid preserved)
        return (int(rng.integers(8, 41)), float(pitch * rng.integers(1, 3)))
    return None


def _apply_boundaries(img, rng, level, ap, pitch):
    """Array-boundary edges: the array terminates and blank field begins.

    On a canvas larger than one tile these are the STREETS between array
    blocks (a finite blank band), which is how a real die tiles its arrays;
    a band touching the canvas edge degenerates to the outer-strip case.
    Count scales with the canvas extent.  (Dummy-fill texture in the field
    is a Phase-2 refinement.)"""
    if level <= 0:
        return
    H, W = img.shape
    n = int(rng.binomial(max(1, int(round(3 * _lin_tiles(max(H, W))))),
                         0.35 * level))
    for _ in range(n):
        axis = 'x' if rng.random() < 0.5 else 'y'
        L = W if axis == 'x' else H
        if _lin_tiles(L) <= 1.0:
            side = 'low' if rng.random() < 0.5 else 'high'
            frac = rng.uniform(0.05, 0.25)
            pos = int(frac * L) if side == 'low' else int((1.0 - frac) * L)
            lo, hi = (0, pos) if side == 'low' else (pos, L)
        else:
            width = int(rng.uniform(1.5, 5.0) * pitch)
            pos = int(rng.uniform(0.02, 0.98) * L)
            lo, hi, side = pos, min(pos + width, L), 'street'
        cut = slice(lo, hi)
        if axis == 'x':
            img[:, cut] = _FIELD
        else:
            img[cut, :] = _FIELD
        ap.append({'type': 'array_boundary', 'axis': axis, 'pos': pos,
                   'side': side})


def render_dram(shape, rng, params=None):
    """DRAM-style ideal layout: horizontal word-lines, vertical bit-lines,
    contact dot at every intersection.  Returns (float32 image in [0,1], meta)."""
    params = params or {}
    H, W = shape
    level = min(max(float(params.get('aperiodic_level', 0.5)), 0.0), 1.0)
    rough = float(params.get('edge_roughness', 0.35))

    pitch_x = float(params.get('pitch_x', rng.uniform(30.0, 60.0)))  # bit-lines (vertical)
    pitch_y = float(params.get('pitch_y', rng.uniform(30.0, 60.0)))  # word-lines (horizontal)
    lw_x = pitch_x * rng.uniform(0.40, 0.55)
    lw_y = pitch_y * rng.uniform(0.40, 0.55)
    r_c = min(pitch_x, pitch_y) * rng.uniform(0.20, 0.30)

    ap = []
    seam_v = _seams(rng, level, H, pitch_x)
    for pos, sh in seam_v:
        ap.append({'type': 'phase_seam', 'family': 'bitline', 'axis': 'y',
                   'pos': pos, 'shift_px': sh})
    seam_h = _seams(rng, level, W, pitch_y)
    for pos, sh in seam_h:
        ap.append({'type': 'phase_seam', 'family': 'wordline', 'axis': 'x',
                   'pos': pos, 'shift_px': sh})
    brk_v = _maybe_break(rng, level, pitch_x)
    brk_h = _maybe_break(rng, level, pitch_y)

    cov_v, cx, xshift = _render_family(W, H, pitch_x, 0.5 * lw_x, rng, rough,
                                       seam_v, brk_v)
    cov_h, cy, yshift = _render_family(H, W, pitch_y, 0.5 * lw_y, rng, rough,
                                       seam_h, brk_h)
    img = cov_v                      # compose in place: large canvases
    np.maximum(img, cov_h.T, out=img)
    del cov_h
    img *= _DRAM_LINE

    if brk_v:
        ap.append({'type': 'subarray_break', 'family': 'bitline', 'axis': 'x',
                   'period_lines': brk_v[0], 'gap_px': brk_v[1],
                   'positions': _break_positions(cx, pitch_x, W)})
    if brk_h:
        ap.append({'type': 'subarray_break', 'family': 'wordline', 'axis': 'y',
                   'period_lines': brk_h[0], 'gap_px': brk_h[1],
                   'positions': _break_positions(cy, pitch_y, H)})

    # contacts at every intersection, displaced consistently with the lattice
    cx_in = cx[(cx >= 0) & (cx < W)]
    cy_in = cy[(cy >= 0) & (cy < H)]
    contacts = [(x + xshift[min(max(int(y), 0), H - 1)],
                 y + yshift[min(max(int(x), 0), W - 1)])
                for x in cx_in for y in cy_in]

    n_def = (int(rng.binomial(int(round(5 * _area_tiles((H, W)))), 0.6 * level))
             if level > 0 else 0)
    broken_rects, particles = [], []
    for _ in range(n_def):
        kind = str(rng.choice(['missing_contact', 'broken_line', 'particle']))
        if kind == 'missing_contact' and contacts:
            x, y = contacts.pop(int(rng.integers(len(contacts))))
            ap.append({'type': 'defect', 'kind': 'missing_contact',
                       'pos': (float(x), float(y))})
        elif kind == 'broken_line':
            if rng.random() < 0.5 and cx_in.size:
                x = float(rng.choice(cx_in))
                y0, ln = rng.uniform(0, H), rng.uniform(1.0, 2.0) * pitch_y
                broken_rects.append((x - 0.5 * lw_x - 1, x + 0.5 * lw_x + 1,
                                     y0, y0 + ln))
                ap.append({'type': 'defect', 'kind': 'broken_line',
                           'family': 'bitline', 'pos': (x, float(y0 + 0.5 * ln))})
            elif cy_in.size:
                y = float(rng.choice(cy_in))
                x0, ln = rng.uniform(0, W), rng.uniform(1.0, 2.0) * pitch_x
                broken_rects.append((x0, x0 + ln,
                                     y - 0.5 * lw_y - 1, y + 0.5 * lw_y + 1))
                ap.append({'type': 'defect', 'kind': 'broken_line',
                           'family': 'wordline', 'pos': (float(x0 + 0.5 * ln), y)})
        else:
            px, py = rng.uniform(0, W), rng.uniform(0, H)
            pr = rng.uniform(0.3, 0.7) * min(pitch_x, pitch_y)
            particles.append((px, py, pr))
            ap.append({'type': 'defect', 'kind': 'particle',
                       'pos': (float(px), float(py)), 'radius_px': float(pr)})

    for x, y in contacts:
        _add_disc(img, x, y, r_c, _DRAM_CONTACT)
    for x0, x1, y0, y1 in broken_rects:
        _paint_rect(img, x0, x1, y0, y1, _FIELD)
    for x, y, r in particles:
        _add_disc(img, x, y, r, _PARTICLE)
    _large_particles(img, rng, level, ap, min(pitch_x, pitch_y), (H, W))
    _apply_boundaries(img, rng, level, ap, min(pitch_x, pitch_y))

    meta = {'style': 'dram', 'pitch_x': pitch_x, 'pitch_y': pitch_y,
            'linewidth_x': lw_x, 'linewidth_y': lw_y, 'contact_radius': r_c,
            'grey_levels': {'field': _FIELD, 'line': _DRAM_LINE,
                            'contact': _DRAM_CONTACT},
            'aperiodic_level': level, 'aperiodic': ap}
    np.clip(img, 0.0, 1.0, out=img)   # already float32; avoid a full-size copy
    return img, meta


def render_finfet(shape, rng, params=None):
    """FinFET-style ideal layout: dense vertical fins crossed by horizontal
    gate bars; crossings brighter than either alone.  Returns (float32, meta)."""
    params = params or {}
    H, W = shape
    level = min(max(float(params.get('aperiodic_level', 0.5)), 0.0), 1.0)
    rough = float(params.get('edge_roughness', 0.35))

    pitch_x = float(params.get('pitch_x', rng.uniform(20.0, 36.0)))          # fins
    lw_fin = pitch_x * rng.uniform(0.30, 0.40)
    pitch_y = float(params.get('pitch_y', pitch_x * rng.uniform(3.0, 5.0)))  # gates
    gate_w = 2.0 * pitch_x * rng.uniform(0.50, 0.70)

    ap = []
    seam_f = _seams(rng, level, H, pitch_x)
    for pos, sh in seam_f:
        ap.append({'type': 'phase_seam', 'family': 'fin', 'axis': 'y',
                   'pos': pos, 'shift_px': sh})
    brk_f = _maybe_break(rng, level, pitch_x)
    seam_g = _seams(rng, 0.5 * level, W, pitch_y)
    for pos, sh in seam_g:
        ap.append({'type': 'phase_seam', 'family': 'gate', 'axis': 'x',
                   'pos': pos, 'shift_px': sh})

    cov_f, fx, fshift = _render_family(W, H, pitch_x, 0.5 * lw_fin, rng, rough,
                                       seam_f, brk_f)
    cov_g, gy, _ = _render_family(H, W, pitch_y, 0.5 * gate_w, rng, rough,
                                  seam_g, None)
    img = cov_f                      # compose in place: crossings = 0.95
    img *= _FIN
    cov_g *= _GATE
    img += cov_g.T
    del cov_g

    if brk_f:
        ap.append({'type': 'subarray_break', 'family': 'fin', 'axis': 'x',
                   'period_lines': brk_f[0], 'gap_px': brk_f[1],
                   'positions': _break_positions(fx, pitch_x, W)})

    fx_in = fx[(fx >= 0) & (fx < W)]
    n_def = (int(rng.binomial(int(round(5 * _area_tiles((H, W)))), 0.6 * level))
             if level > 0 else 0)
    for _ in range(n_def):
        kind = str(rng.choice(['broken_fin', 'particle']))
        if kind == 'broken_fin' and fx_in.size:
            x = float(rng.choice(fx_in))
            y0, ln = rng.uniform(0, H), rng.uniform(1.0, 3.0) * pitch_x
            _paint_rect(img, x - 0.5 * lw_fin - 1, x + 0.5 * lw_fin + 1,
                        y0, y0 + ln, _FIELD)
            ap.append({'type': 'defect', 'kind': 'broken_fin',
                       'pos': (x, float(y0 + 0.5 * ln))})
        else:
            px, py = rng.uniform(0, W), rng.uniform(0, H)
            pr = rng.uniform(0.3, 0.6) * pitch_x
            _add_disc(img, px, py, pr, _PARTICLE)
            ap.append({'type': 'defect', 'kind': 'particle',
                       'pos': (float(px), float(py)), 'radius_px': float(pr)})
    _large_particles(img, rng, level, ap, pitch_x, (H, W))
    _apply_boundaries(img, rng, level, ap, pitch_x)

    meta = {'style': 'finfet', 'pitch_x': pitch_x, 'pitch_y': pitch_y,
            'fin_width': lw_fin, 'gate_width': gate_w,
            'grey_levels': {'field': _FIELD, 'fin': _FIN, 'gate': _GATE,
                            'crossing': _FIN + _GATE},
            'aperiodic_level': level, 'aperiodic': ap}
    np.clip(img, 0.0, 1.0, out=img)   # already float32; avoid a full-size copy
    return img, meta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for fn in (render_dram, render_finfet):
        img, meta = fn((512, 512), rng, {'aperiodic_level': 1.0})
        assert img.dtype == np.float32
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        assert len(meta['aperiodic']) > 0
        print(f"{meta['style']}: shape={img.shape} "
              f"range=({img.min():.2f},{img.max():.2f}) "
              f"pitch=({meta['pitch_x']:.1f},{meta['pitch_y']:.1f}) "
              f"aperiodic features={len(meta['aperiodic'])}")
        img0, meta0 = fn((512, 512), rng, {'aperiodic_level': 0.0})
        assert meta0['aperiodic'] == [], "level=0 must be a pure lattice"
    print("layout self-check OK")
