"""Drift-Sense — navigation-error recovery.

    python localize.py --ref ref.png --search search.png

Prints exactly one line to stdout: the (x, y) pixel centre of the region in
the search image where the reference pattern appears.

    512.34, 487.91

Optional:
    --json          append one line of JSON diagnostics
    --viz out.png   write an overlay image.  This is the only file the program
                    itself writes; CPython still writes core/__pycache__/*.pyc
                    on import, which PYTHONDONTWRITEBYTECODE=1 suppresses.
    --scale-prior   prior on the magnification ratio (default 0.1 == "~10x").
                    This is a PRIOR for the search, never a hardcoded answer:
                    the scale is estimated from the images by core.lattice.

This program never raises.  If any stage fails it degrades to plain
TM_CCOEFF_NORMED at the prior scale, and if that fails it reports the centre
of the search image.  A wrong answer scores more than a traceback.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    import numpy as np
except ImportError as _import_exc:         # a broken install must not traceback
    # Bind the message NOW: Python deletes the `except ... as` name when the
    # block ends, so a closure over it would be a latent NameError.
    _IMPORT_ERROR = str(_import_exc)

    def _no_deps(argv=None):
        """Report the missing dependency loudly on stderr, but still emit a
        coordinate on stdout so a batch harness parses a row rather than a
        crash.  Exit code is non-zero so the failure is not silent."""
        sys.stderr.write(
            f"localize.py: missing dependency ({_IMPORT_ERROR}). "
            f"Install with: {sys.executable} -m pip install -r "
            f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')}\n")
        x = y = 0.0
        try:                              # best effort: centre of the search
            from PIL import Image
            if argv is None:
                argv = sys.argv[1:]
            if '--search' in argv:
                with Image.open(argv[argv.index('--search') + 1]) as im:
                    x, y = (im.size[0] - 1) / 2.0, (im.size[1] - 1) / 2.0
        except Exception:
            pass
        print(f"{x:.2f}, {y:.2f}")
        return 2

    if __name__ == '__main__':
        sys.exit(_no_deps())
    raise

SUPPORTED = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def load_grey(path):
    """Read an image as 8-bit greyscale.  Handles colour, alpha, 16-bit and
    float inputs; never assumes a size."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:                       # unusual encoding: try Pillow
        from PIL import Image
        with Image.open(path) as im:
            img = np.array(im.convert('L'))
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]
    if img.dtype != np.uint8:             # 16-bit or float -> robust 8-bit
        a = img.astype(np.float64)
        lo, hi = np.percentile(a, [0.5, 99.5])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi <= lo:
            hi = lo + 1.0
        img = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"unusable image: {path}")
    return np.ascontiguousarray(img)


def _sane_A(A, scale_prior, window=4.0):
    """Reject a transform that is non-finite, reflecting, wildly anisotropic
    or far from the prior — a bad geometry produces a nonsense template."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (2, 2) or not np.isfinite(A).all():
        return False
    det = float(np.linalg.det(A))
    if det <= 0:
        return False
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[1] <= 1e-9 or sv[0] / sv[1] > 1.5:
        return False
    scale = float(np.sqrt(sv[0] * sv[1]))
    return scale_prior / window < scale < scale_prior * window


def fallback_match(ref_u8, search_u8, scale_prior):
    """Plain TM_CCOEFF_NORMED at the prior scale."""
    hs, ws = search_u8.shape
    s = float(scale_prior)
    tw = max(8, int(round(ref_u8.shape[1] * s)))
    th = max(8, int(round(ref_u8.shape[0] * s)))
    # the template must fit inside the search image
    if tw > ws or th > hs:
        shrink = min(ws / float(tw), hs / float(th)) * 0.9
        tw, th = max(8, int(tw * shrink)), max(8, int(th * shrink))
    tmpl = cv2.resize(ref_u8, (tw, th), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(search_u8.astype(np.float32),
                            tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    return loc[0] + (tw - 1) / 2.0, loc[1] + (th - 1) / 2.0


def localize(ref_u8, search_u8, scale_prior=0.1, lam=None, k=None,
             timings=None, return_score_map=False, diag=None):
    """Full pipeline with graceful degradation.  Returns (x, y, diag).

    lam / k override core.correlate.LAM and core.resolve._K (evaluate.py
    sweeps them); timings, if a dict, is filled with per-stage seconds.

    diag, if a dict, is filled IN PLACE as the pipeline progresses, so a
    caller that catches an exception can still read how far it got.  Returning
    the dict is not enough for that: on a raise there is no return value, and
    the caller's own name would still be bound to whatever it held before.
    """
    from core.correlate import (LAM, build_template, combine, match_psf,
                                score_maps)
    from core.lattice import estimate_transform
    from core.preprocess import anscombe, normalise, prep
    from core.refine import refine, rescore_candidates
    from core.resolve import _K, resolve, resolve_ranked

    lam = LAM if lam is None else float(lam)
    k = _K if k is None else float(k)

    def _t(name, t0):
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + (time.time() - t0)
        return time.time()

    diag = {} if diag is None else diag
    diag.update(method='lattice', stage='start')
    t0 = time.time()
    search = prep(search_u8)
    ref_prepped = prep(ref_u8)
    t0 = _t('prep', t0)

    diag['stage'] = 'estimate_transform'
    A, tdiag = estimate_transform(ref_prepped, search, scale_prior=scale_prior)
    t0 = _t('estimate_transform', t0)
    if not _sane_A(A, scale_prior):
        # keep going with the prior rather than a nonsense geometry
        A = float(scale_prior) * np.eye(2)
        diag['transform_rejected'] = True
    diag['transform'] = {key: tdiag[key] for key in
                         ('method', 'scale', 'rotation_deg', 'anisotropy',
                          'sharpness', 'match_score')
                         if key in tdiag}

    diag['stage'] = 'build_template'
    tmpl = build_template(anscombe(ref_u8), A)
    if min(tmpl.shape) < 8 or tmpl.shape[0] > search.shape[0] \
            or tmpl.shape[1] > search.shape[1]:
        raise ValueError(f"template {tmpl.shape} unusable against "
                         f"search {search.shape}")
    tmpl = match_psf(normalise(tmpl), search)
    t0 = _t('build_template', t0)

    diag['stage'] = 'score_maps'
    cache = {}
    S_full, S_res = score_maps(tmpl, search, cache=cache)
    S = combine(S_full, S_res, lam=lam)
    t0 = _t('score_maps', t0)

    diag['stage'] = 'resolve'
    px, py, rinfo = resolve(S, search.shape, k=k)
    t0 = _t('resolve', t0)
    diag['resolve'] = {key: rinfo[key] for key in
                       ('n_peaks', 'family_size', 'confidence',
                        'tie_break_used', 'pitch', 'best_score',
                        'second_score')}

    diag['stage'] = 'rescore'
    x, y = float(px), float(py)
    try:
        (bx, by), ranked, sinfo = rescore_candidates(
            S, anscombe(ref_u8), search, A, template=tmpl,
            pitch=rinfo['pitch'], tmpl_res=cache.get('res_t'),
            search_res=cache.get('res_s'))
        diag['rescore'] = sinfo
        # The centre rule is a HARD SPEC RULE, so it must act on the scores
        # the pipeline actually decides with.  Re-apply it to the rescored
        # candidates rather than leaving the coarse-map tie-break to be
        # overwritten here (which would make it dead code).
        x, y, rinfo2 = resolve_ranked([(t[0], t[1], t[2]) for t in ranked],
                                      search.shape, k=k)
        diag['resolve'] = {key: rinfo2[key] for key in
                           ('n_peaks', 'family_size', 'confidence',
                            'tie_break_used', 'best_score', 'second_score')}
        diag['resolve']['pitch'] = rinfo['pitch']
        diag['resolve_coarse'] = {'x': float(px), 'y': float(py),
                                  'tie_break_used': bool(
                                      rinfo['tie_break_used'])}
    except Exception as exc:                       # rescoring is optional
        diag['rescore_error'] = f"{type(exc).__name__}: {exc}"
    t0 = _t('rescore', t0)

    diag['stage'] = 'refine'
    try:
        rx, ry, fdiag = refine(S, (x, y), anscombe(ref_u8), search, A,
                               template=tmpl)
        diag['refine'] = {key: fdiag[key] for key in
                          ('method', 'ecc_rejected', 'parabolic_rejected',
                           'ecc_cc', 'ecc_shift_px', 'confidence')
                          if key in fdiag}
        x, y = rx, ry
    except Exception as exc:                       # refinement is optional
        diag['refine_error'] = f"{type(exc).__name__}: {exc}"
    t0 = _t('refine', t0)

    diag['stage'] = 'done'
    try:
        # Reject option only.  This never moves (x, y) -- it annotates it.
        # A missing or unreadable model simply omits the field.
        from core.confidence import score as _confidence
        p = _confidence(diag)
        if p is not None:
            diag['p_correct'] = p
    except Exception as exc:                       # confidence is optional
        diag['confidence_error'] = f"{type(exc).__name__}: {exc}"

    if return_score_map:          # opt-in: keeps --json output small
        diag['score_map'] = S
    return float(x), float(y), diag


def run(ref_path, search_path, scale_prior=0.1, viz=None):
    t0 = time.time()
    diag = {}
    try:
        ref_u8 = load_grey(ref_path)
        search_u8 = load_grey(search_path)
    except Exception as exc:
        # cannot even read the inputs: there is no image to report about
        return None, None, {'method': 'load_failed',
                            'error': f"{type(exc).__name__}: {exc}"}
    hs, ws = search_u8.shape
    try:
        # pass diag IN so the stage survives an exception (see localize())
        x, y, diag = localize(ref_u8, search_u8, scale_prior, diag=diag)
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError("non-finite result")
    except Exception as exc:
        failed_at = diag.get('stage', 'unknown')
        try:
            x, y = fallback_match(ref_u8, search_u8, scale_prior)
            diag = {'method': 'fallback_ccoeff', 'failed_stage': failed_at,
                    'error': f"{type(exc).__name__}: {exc}"}
        except Exception as exc2:
            x, y = (ws - 1) / 2.0, (hs - 1) / 2.0
            diag = {'method': 'centre_of_search', 'failed_stage': failed_at,
                    'error': f"{type(exc).__name__}: {exc}",
                    'fallback_error': f"{type(exc2).__name__}: {exc2}"}
    # the answer must lie inside the search image
    x = float(min(max(x, 0.0), ws - 1))
    y = float(min(max(y, 0.0), hs - 1))
    diag.update(search_shape=[hs, ws], ref_shape=list(ref_u8.shape),
                elapsed_s=round(time.time() - t0, 4), x=x, y=y)

    if viz:
        try:
            over = cv2.cvtColor(search_u8, cv2.COLOR_GRAY2BGR)
            ix, iy = int(round(x)), int(round(y))
            r = max(12, min(hs, ws) // 25)
            cv2.line(over, (ix - r, iy), (ix + r, iy), (0, 0, 255), 2)
            cv2.line(over, (ix, iy - r), (ix, iy + r), (0, 0, 255), 2)
            cv2.circle(over, (ix, iy), r, (0, 255, 255), 2)
            cv2.putText(over, f"{x:.2f}, {y:.2f}", (max(ix - r, 5),
                        max(iy - r - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # imwrite RETURNS False on failure (bad directory, unknown
            # extension); it does not raise, so the result must be checked or
            # the diagnostics claim a file that was never written.
            if cv2.imwrite(viz, over):
                diag['viz'] = viz
            else:
                diag['viz_error'] = f"cv2.imwrite returned False for {viz!r}"
        except Exception as exc:
            diag['viz_error'] = f"{type(exc).__name__}: {exc}"
    return x, y, diag


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Locate a high-magnification SEM reference inside a "
                    "lower-magnification search image.")
    p.add_argument('--ref', required=True, help=f"reference image {SUPPORTED}")
    p.add_argument('--search', required=True, help="search image")
    p.add_argument('--json', action='store_true',
                   help="append one line of JSON diagnostics")
    p.add_argument('--viz', default=None, metavar='OUT.png',
                   help="write an overlay image (the only file the program "
                        "itself writes)")
    p.add_argument('--scale-prior', type=float, default=0.1,
                   help="prior on the magnification ratio (default 0.1)")
    args = p.parse_args(argv)

    x, y, diag = run(args.ref, args.search, args.scale_prior, args.viz)
    # THE contract: a coordinate on stdout FIRST, always, on every path.  A
    # batch harness parses column 1, so an unreadable input must still yield a
    # parseable row -- with a non-zero exit code so the failure is not silent.
    failed = x is None
    if failed:
        x = y = 0.0
    print(f"{x:.2f}, {y:.2f}")
    if args.json:
        print(json.dumps(diag, default=str))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
