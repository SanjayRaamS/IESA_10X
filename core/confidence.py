"""Confidence estimation for Drift-Sense (Phase 11).

A logistic regression over six peak-surface features predicts whether a
prediction is CORRECT (error <= 5 px), not where the answer is.  It is a
reject option, never a matcher: `localize.py` reports the same coordinate
whether this model is present, absent or wrong.

    p = sigmoid( ((f - mu) / sd) . w + b )

Training lives in `train_confidence.py` and uses scikit-learn; nothing here
imports it.  The fitted model is six coefficients, an intercept and a
standardiser, stored as a ~2 KB .npz, so inference is one dot product and the
runtime dependency set stays numpy-only (CLAUDE.md hard constraint).

The features are deliberately all SCALE-FREE — ratios, counts and correlation
coefficients, never raw scores or pixel sizes.  A feature measured in score
units would not transfer between images of different dose or contrast, and the
whole point of a reject option is that it transfers.

    peak_ratio        second-best peak / best peak, on the combined score
                      surface.  ~1 means an ambiguous lattice, << 1 means one
                      cell won outright.  THE headline feature.
    resolve_conf      (best - best-outside-family) / MAD(S).  How far the
                      winner stands above its rivals in units of the surface's
                      own oscillation.
    family_size       how many peaks were statistically tied.  1 is decisive;
                      larger means the centre rule had to choose.
    residual_zncc     agreement between the lattice-notched reference residual
                      and the search residual at the chosen cell.  This is the
                      aperiodic evidence -- the thesis's own signal.
    ecc_cc            final ECC correlation from refinement.  High means the
                      warped reference genuinely sits on that patch.
    lattice_sharpness mean curvature of the matched reciprocal-lattice peaks.
                      Sharp peaks mean the geometry estimate is trustworthy,
                      so a wrong answer is more likely a cell-selection error
                      than a bad warp.
"""

import os

import numpy as np

FEATURES = ('peak_ratio', 'resolve_conf', 'family_size',
            'residual_zncc', 'ecc_cc', 'lattice_sharpness')

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'confidence_model.npz')

# error at or below this is "correct" -- the tolerance the brief cares about
CORRECT_PX = 5.0

_CONF_CLIP = 30.0        # keeps exp() finite without changing any decision


def _finite(x, default=0.0):
    x = float(x) if x is not None else default
    return x if np.isfinite(x) else default


def extract_features(diag):
    """Six features from a localize() diagnostics dict.

    Missing stages are the normal case, not an error: rescore and refine are
    both optional and `localize.py` degrades without them.  A missing feature
    becomes a neutral value rather than a NaN, so the model still produces a
    probability for a degraded run instead of refusing to score it.
    """
    res = diag.get('resolve') or {}
    rsc = diag.get('rescore') or {}
    ref = diag.get('refine') or {}
    tra = diag.get('transform') or {}

    best = _finite(res.get('best_score'), 0.0)
    second = _finite(res.get('second_score'), 0.0)
    # scores can straddle zero, so a raw ratio is not safe; anchor on
    # magnitude and clamp to [0, 1] where 1 == "the rivals are as good"
    denom = max(abs(best), 1e-6)
    peak_ratio = float(np.clip(abs(second) / denom, 0.0, 1.0))

    conf = _finite(res.get('confidence'), 0.0)
    if not np.isfinite(conf):
        conf = 0.0
    conf = float(np.clip(conf, 0.0, 50.0))     # 'inf' when nothing is outside

    return np.array([
        peak_ratio,
        conf,
        float(res.get('family_size', 1) or 1),
        _finite(rsc.get('best_residual_zncc'), 0.0),
        _finite(rsc.get('best_ecc_cc'), _finite(ref.get('ecc_cc'), 0.0)),
        _finite(tra.get('sharpness'), 0.0),
    ], dtype=np.float64)


def load_model(path=None):
    """Return a model dict, or None if there is no usable model on disk.

    Never raises: a missing or corrupt model must degrade to "no confidence
    reported", exactly like a missing rescore stage.
    """
    path = path or DEFAULT_MODEL
    try:
        with np.load(path, allow_pickle=False) as z:
            model = {'w': z['w'].astype(np.float64),
                     'b': float(z['b']),
                     'mu': z['mu'].astype(np.float64),
                     'sd': z['sd'].astype(np.float64),
                     'features': tuple(str(s) for s in z['features'])}
    except Exception:
        return None
    if model['features'] != FEATURES or model['w'].shape != (len(FEATURES),):
        return None                      # a model for different features
    return model


def predict_proba(feats, model):
    """P(correct) for one feature vector.  Pure numpy, no sklearn at runtime."""
    x = (np.asarray(feats, dtype=np.float64) - model['mu']) / model['sd']
    z = float(np.dot(x, model['w']) + model['b'])
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -_CONF_CLIP, _CONF_CLIP))))


def score(diag, model=None):
    """P(correct) straight from a localize() diag, or None without a model."""
    model = model if model is not None else load_model()
    if model is None:
        return None
    return predict_proba(extract_features(diag), model)


if __name__ == "__main__":
    m = load_model()
    if m is None:
        print("confidence: no model at", DEFAULT_MODEL,
              "-- run train_confidence.py")
    else:
        print(f"confidence: model over {len(m['features'])} features")
        for name, w in sorted(zip(m['features'], m['w']),
                              key=lambda kv: -abs(kv[1])):
            print(f"    {name:18s} w={w:+.3f}")
        print(f"    {'(intercept)':18s} b={m['b']:+.3f}")

        # a decisive surface must score above an ambiguous one, whatever the
        # fitted signs are -- this is the only invariant worth asserting here
        decisive = {'resolve': {'best_score': 1.0, 'second_score': 0.2,
                                'confidence': 8.0, 'family_size': 1},
                    'rescore': {'best_residual_zncc': 0.5,
                                'best_ecc_cc': 0.95},
                    'transform': {'sharpness': 2.0}}
        ambiguous = {'resolve': {'best_score': 1.0, 'second_score': 0.99,
                                 'confidence': 0.1, 'family_size': 6},
                     'rescore': {'best_residual_zncc': 0.02,
                                 'best_ecc_cc': 0.55},
                     'transform': {'sharpness': 2.0}}
        pd, pa = score(decisive, m), score(ambiguous, m)
        print(f"    decisive surface  P(correct)={pd:.3f}")
        print(f"    ambiguous surface P(correct)={pa:.3f}")
        assert pd > pa, "decisive surface must not score below an ambiguous one"
        print("confidence self-check OK")
