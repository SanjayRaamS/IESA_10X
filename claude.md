# Drift-Sense — Navigation-Error Recovery

## What we're building
Given a Reference SEM image (high magnification) and a 1000x1000 Search image
(10x lower magnification), output the (x, y) pixel centre of the region in the
Search image where the Reference pattern appears, shrunk ~10x.

If several regions match equally well, return the one CLOSEST TO THE CENTRE of
the Search image. This is a hard spec rule, not a heuristic. It must be an
explicit, tested code path — never an accidental consequence of argmax.

## Core algorithmic thesis (do not deviate without asking)
Semiconductor layouts are periodic. The periodic component of an image carries
ZERO position information but PERFECT scale/rotation information. The aperiodic
residual (array edges, tile seams, dummy-fill boundaries, defects) carries the
only usable position information.

Therefore:
  - Periodic component -> solve geometry (scale + rotation) analytically via FFT.
  - Aperiodic residual  -> break the periodic tie and pick the correct unit cell.
Classical template matching fails because it mixes the two and the periodic
energy drowns the residual.

## Hard constraints
- CPU only. No GPU, no torch, no downloaded weights, no network calls at runtime.
- numpy, scipy, opencv-python (NOT contrib), Pillow, matplotlib. Nothing else
  unless you ask first.
- Target < 300 ms per pair single-core. Accuracy beats speed in every tradeoff.
- Never hardcode the 10x factor. Always ESTIMATE scale from the images. The test
  set has scale variation.
- All randomness through np.random.default_rng(seed). No global np.random.
- Every module has a `if __name__ == "__main__":` self-check.

## Working style
- State assumptions before implementing. If two interpretations exist, present
  both, don't pick silently.
- Minimum code that solves the problem. No speculative abstraction, no config
  systems nobody asked for.
- Each phase ends with its verify gate PASSING and printed. If a gate fails,
  stop and report — do not proceed to the next phase.
- Match existing style once files exist. Don't refactor working code.

## Repo layout
drift-sense/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── REFERENCES.md
├── generate_dataset.py      # standalone entry point
├── localize.py              # standalone entry point — THE scored file
├── evaluate.py              # standalone entry point
├── core/
│   ├── __init__.py
│   ├── layout.py            # ideal binary layouts
│   ├── sem.py               # SEM forward imaging model
│   ├── preprocess.py
│   ├── lattice.py           # FFT scale/rotation
│   ├── correlate.py         # dual-channel score maps
│   ├── resolve.py           # ambiguity resolution + centre rule
│   └── refine.py            # subpixel + ECC
├── tests/
└── data/
