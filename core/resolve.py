"""Ambiguity resolution and the centre rule for Drift-Sense (Phase 7).

A periodic layout produces a score surface with many statistically tied peaks
(Gate 6b measures ~50 of them within 2% of the maximum on a pure lattice).
Picking the argmax among ties is a coin flip.  The brief's rule is explicit and
must be an explicit, tested code path: when several regions match equally well,
return the one CLOSEST TO THE CENTRE of the search image.

`tie_break_used` records when that rule actually decided the answer.  The test
set is guaranteed to contain at least one deliberately ambiguous highly-periodic
region, and this flag is the evidence that it was handled by design.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_SEP_FRAC = 0.7        # NMS separation as a fraction of the tile pitch
_PITCH_RANGE = (3.0, 40.0)
_PITCH_FFT = 512       # centre crop used for the pitch FFT
_SIGMA_FLOOR = 1e-12   # keeps exactly-tied peaks in one family
# Family width in sigma.  Swept over the brief's grid {1.0,1.5,2.0,3.0} plus
# {0.05,0.1,0.5}; k>=1 is decisively worse on both the training and the
# fresh-seed set (it fires the tie-break on most pairs and overrides correct
# answers), k<=0.5 is flat.  See resolve() for the measured table.
_K = 0.1


def _estimate_pitch(S, valid):
    """Dominant period of the score surface, in search pixels.

    The score map inherits the layout's periodicity, so its own dominant
    Fourier period is the tile pitch — no need to thread the lattice estimate
    through from Phase 5."""
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return _PITCH_RANGE[0]
    sub = S[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sub = np.nan_to_num(np.asarray(sub, dtype=np.float64),
                        neginf=0.0, posinf=0.0)
    h, w = sub.shape
    if min(h, w) < 8:
        return _PITCH_RANGE[0]
    # the pitch only sets an NMS radius, so a centre crop is plenty and keeps
    # the FFT off the full-size map
    if min(h, w) > _PITCH_FFT:
        cy0, cx0 = (h - _PITCH_FFT) // 2, (w - _PITCH_FFT) // 2
        sub = sub[cy0:cy0 + _PITCH_FFT, cx0:cx0 + _PITCH_FFT]
        h, w = sub.shape
    sub = (sub - sub.mean()) * np.outer(np.hanning(h), np.hanning(w))
    F = np.abs(np.fft.fftshift(np.fft.fft2(sub)))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    fy, fx = (yy - cy) / h, (xx - cx) / w
    r = np.hypot(fx, fy)
    F[r < 1.0 / _PITCH_RANGE[1]] = 0.0      # ignore periods above the range
    F[r > 1.0 / _PITCH_RANGE[0]] = 0.0
    if not np.any(F > 0):
        return _PITCH_RANGE[0]
    i = int(np.argmax(F))
    period = 1.0 / max(r.flat[i], 1e-9)
    return float(np.clip(period, *_PITCH_RANGE))


def _local_maxima(S, valid, sep):
    """Non-maximum-suppressed local maxima, strongest first.

    Ordering is a total order (score desc, then y, then x) so the result is
    deterministic even when scores tie exactly."""
    k = int(2 * max(1, int(round(sep))) + 1)
    Sf = np.where(valid, S, -np.inf).astype(np.float32)
    mx = cv2.dilate(Sf, np.ones((k, k), np.float32))
    ys, xs = np.nonzero(valid & (Sf >= mx - 1e-12))
    scores = np.asarray(S, dtype=np.float64)[ys, xs]
    order = np.lexsort((xs, ys, -scores))   # last key is primary
    return xs[order], ys[order], scores[order]


def _apply_centre_rule(xs, ys, scores, sigma, k, search_shape):
    """Shared family + centre-rule core.

    Returns (index picked, family indices, tie_break_used, distances)."""
    H, W = search_shape
    in_family = (scores[0] - scores) < k * sigma
    in_family[0] = True                 # the best is always in its own family
    fam = np.nonzero(in_family)[0]
    # Pixel indices run 0..W-1, so the centre of the image is (W-1)/2, NOT
    # W/2.  The half-pixel matters only as correctness of a hard spec rule
    # (rival peaks sit a pitch apart), but localize.py's own centre fallback
    # already uses (W-1)/2 and the two must not disagree.
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dist = np.hypot(xs[fam] - cx, ys[fam] - cy)
    if fam.size == 1:
        return fam[0], fam, False, dist
    # nearest the search-image centre, with a total order (distance, then
    # score desc, then y, then x) so exact ties stay deterministic
    j = sorted(range(fam.size),
               key=lambda t: (dist[t], -scores[fam[t]], ys[fam[t]], xs[fam[t]]))[0]
    return fam[j], fam, True, dist


def resolve_ranked(ranked, search_shape, k=_K):
    """Apply the centre rule to an EXPLICIT ranked candidate list.

    `ranked` is a sequence of (score, x, y), best first — typically the output
    of core.refine.rescore_candidates.  The brief's rule is a hard spec rule,
    so it has to act on the scores the pipeline actually decides with: once
    candidates are re-scored with ECC and aperiodic evidence, the tie-break
    must be re-applied to THOSE scores, otherwise the rule is computed on the
    coarse map and then discarded.

    Returns (x, y, info) with the same info keys as resolve().
    """
    ranked = list(ranked)
    if not ranked:
        raise ValueError("empty candidate list")
    scores = np.array([r[0] for r in ranked], dtype=float)
    xs = np.array([r[1] for r in ranked], dtype=float)
    ys = np.array([r[2] for r in ranked], dtype=float)
    order = np.lexsort((xs, ys, -scores))
    scores, xs, ys = scores[order], xs[order], ys[order]
    sigma = max(1.4826 * float(np.median(np.abs(scores - np.median(scores)))),
                _SIGMA_FLOOR)
    pick, fam, tie, dist = _apply_centre_rule(xs, ys, scores, sigma, k,
                                              search_shape)
    outside = np.nonzero(~np.isin(np.arange(scores.size), fam))[0]
    info = {'n_peaks': int(scores.size), 'family_size': int(fam.size),
            'confidence': (float((scores[0] - scores[outside[0]]) / sigma)
                           if outside.size else float('inf')),
            'sigma': sigma, 'pitch': float('nan'),
            'tie_break_used': bool(tie),
            'peak_scores': [float(s) for s in scores[fam]],
            'distances_to_centre': [float(d) for d in dist],
            'best_score': float(scores[0]),
            'second_score': (float(scores[1]) if scores.size > 1
                             else float('nan'))}
    if tie:
        log.warning("TIE-BREAK USED (rescored candidates): %d of %d tied; "
                    "centre rule selected (%.1f, %.1f) at %.1f px from centre",
                    fam.size, scores.size, float(xs[pick]), float(ys[pick]),
                    float(np.hypot(xs[pick] - (search_shape[1] - 1) / 2.0,
                                   ys[pick] - (search_shape[0] - 1) / 2.0)))
    return float(xs[pick]), float(ys[pick]), info


def resolve(S, search_shape, k=_K, pitch=None):
    """Pick the answer from a score surface, applying the centre rule.

    Returns (x, y, info).  info carries n_peaks, family_size, confidence,
    peak_scores, tie_break_used and distances_to_centre (the last two lists
    describe the tied family, strongest first).

    On `k`: the tie-break must fire on genuinely ambiguous surfaces and stay
    out of the way otherwise — firing it on a resolved surface replaces a
    correct argmax with a centre-biased guess.  Measured over the 40-pair set
    (fraction of pairs where the tie-break fired, by aperiodic_level):

        k     ambiguous (ap<0.06)   resolved (top-8 ap)   within 15 px
        0.05        5/8                   1/8                 45%
        0.10        7/8                   2/8                 48%
        0.35        8/8                   5/8                 38%
        2.00        8/8                   8/8                  5%

    Hence the default below rather than the ~2 that a Gaussian reading of
    "k sigma" suggests.  sigma here is MAD over the whole score surface, and
    that surface OSCILLATES with the lattice: its spread measures the
    periodic swing between cells, which is an order of magnitude larger than
    the height difference between rival peaks.  k is the caller's knob and is
    re-checked in Phase 9.
    """
    S = np.asarray(S)
    H, W = search_shape
    valid = np.isfinite(S)
    if not valid.any():
        raise ValueError("score map has no finite entries")

    if pitch is None:
        pitch = _estimate_pitch(S, valid)
    xs, ys, scores = _local_maxima(S, valid, _SEP_FRAC * pitch)

    # robust noise scale of the score surface
    v = S[valid]
    med = float(np.median(v))
    sigma = max(1.4826 * float(np.median(np.abs(v - med))), _SIGMA_FLOOR)

    best = 0
    cxr, cyr = (W - 1) / 2.0, (H - 1) / 2.0
    pick, fam, tie_break_used, dist = _apply_centre_rule(
        xs, ys, scores, sigma, k, search_shape)
    in_family = np.isin(np.arange(scores.size), fam)
    outside = np.nonzero(~in_family)[0]
    confidence = (float((scores[best] - scores[outside[0]]) / sigma)
                  if outside.size else float('inf'))

    info = {'n_peaks': int(xs.size), 'family_size': int(fam.size),
            'confidence': confidence, 'sigma': sigma, 'pitch': float(pitch),
            'tie_break_used': bool(tie_break_used),
            'peak_scores': [float(s) for s in scores[fam]],
            'distances_to_centre': [float(d) for d in dist],
            'best_score': float(scores[best]),
            # raw second-best peak: callers form their own peak-to-second
            # statistic (scores can straddle zero, so the ratio needs a guard
            # that belongs with the consumer, not here)
            'second_score': (float(scores[1]) if scores.size > 1
                             else float('nan'))}
    if tie_break_used:
        log.warning("TIE-BREAK USED: %d peaks within %.1f sigma of the best "
                    "score; centre rule selected (%.1f, %.1f) at %.1f px from "
                    "centre (nearest rival %.1f px)", fam.size, k,
                    float(xs[pick]), float(ys[pick]),
                    float(np.hypot(xs[pick] - cxr, ys[pick] - cyr)),
                    float(np.sort(dist)[1]) if dist.size > 1 else float('nan'))
    return float(xs[pick]), float(ys[pick]), info


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    def blob(S, x, y, amp=1.0, sigma=3.0):
        h, w = S.shape
        yy, xx = np.mgrid[0:h, 0:w]
        S += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    S = np.zeros((1000, 1000), dtype=np.float32)
    blob(S, 300, 300)
    blob(S, 520, 510)
    S[300, 300] = S[510, 520] = 1.0
    x, y, info = resolve(S, (1000, 1000), pitch=10.0)
    assert (x, y) == (520.0, 510.0), (x, y)
    assert info['tie_break_used']
    print(f"resolve: tied peaks -> ({x:.0f},{y:.0f}) family={info['family_size']} "
          f"confidence={info['confidence']:.1f} sigma")

    S2 = np.zeros((1000, 1000), dtype=np.float32)
    blob(S2, 700, 500)
    S2[500, 700] = 1.0
    x2, y2, i2 = resolve(S2, (1000, 1000), pitch=10.0)
    assert not i2['tie_break_used'] and (x2, y2) == (700.0, 500.0)
    print(f"resolve: single peak -> ({x2:.0f},{y2:.0f}) "
          f"tie_break_used={i2['tie_break_used']}")
    print("resolve self-check OK")
