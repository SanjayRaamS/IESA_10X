"""Drift-Sense demo reel — side-by-side animation for the optional video.

    python make_demo.py --data data/train --out results/demo.gif

Runs the real pipeline (the same code path as localize.py) over a handful of
pairs and animates each one as a three-beat reveal:

    beat 1  "here is what the tool is given"     reference | search | score
    beat 2  "here is where it says the reference is"        + prediction
    beat 3  "here is the truth"                             + truth and error

Pairs are chosen to tell an honest story, not a flattering one: by default the
reel is mostly successes plus the WORST pair in the set, so the failure mode is
on screen rather than edited out.  --pairs overrides the selection.

Writes a GIF via Pillow, and an MP4 via cv2.VideoWriter when --mp4 is given
(mp4v is in the base opencv-python wheel, so this adds no dependency).
No output is written outside --out's directory (CPython's own __pycache__
aside).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import evaluate as E

FIGSIZE = (16.0, 5.6)      # fixed so every frame is byte-identical in size
DPI = 80
TRUE_C = '#2ecc71'
PRED_C = '#e74c3c'


def pick_pairs(rows, n):
    """Successes plus the honest worst case.  Never silently all-successes."""
    order = sorted(rows, key=lambda r: r['error_px'])
    if n >= len(order):
        return order
    chosen = order[:max(n - 1, 1)]            # the best n-1 ...
    if n > 1:
        chosen.append(order[-1])              # ... and the worst, always
    return chosen


def footprint(rec_row, ref_shape):
    """Corners of the reference's footprint in search pixels, as drawn by the
    estimated (not the true) geometry — so a wrong box is visibly wrong."""
    t = rec_row.get('transform') or {}
    s = float(t.get('scale', 0.1))
    rot = np.deg2rad(float(t.get('rotation_deg', 0.0)))
    h, w = ref_shape
    c, sn = np.cos(rot), np.sin(rot)
    half = np.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]]) * 0.5 * s
    R = np.array([[c, -sn], [sn, c]])
    pts = half @ R.T
    return pts[:, 0] + rec_row['pred_x'], pts[:, 1] + rec_row['pred_y']


def render_beat(ref_u8, search_u8, rec_row, beat):
    """One RGB frame.  beat 0 = given, 1 = prediction, 2 = truth revealed."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)

    axes[0].imshow(ref_u8, cmap='gray')
    axes[0].set_title(f"reference — {rec_row['style']}, "
                      f"aperiodic={rec_row['aperiodic_level']:.2f}", fontsize=11)

    axes[1].imshow(search_u8, cmap='gray')
    axes[1].set_title("search (10x lower magnification)", fontsize=11)

    S = rec_row.get('score_map')
    if S is not None:
        axes[2].imshow(S, cmap='magma')
        axes[2].set_title("aperiodic score surface", fontsize=11)
    else:
        axes[2].axis('off')

    if beat >= 1:
        fx, fy = footprint(rec_row, ref_u8.shape)
        axes[1].plot(fx, fy, '-', color=PRED_C, lw=1.6, alpha=0.9)
        axes[1].plot([rec_row['pred_x']], [rec_row['pred_y']], 'x', ms=15,
                     color=PRED_C, mew=2.6)
        axes[1].set_title(f"predicted  ({rec_row['pred_x']:.1f}, "
                          f"{rec_row['pred_y']:.1f})  in "
                          f"{rec_row['wall_s']*1000:.0f} ms", fontsize=11)
    if beat >= 2:
        axes[1].plot([rec_row['true_x']], [rec_row['true_y']], 'o', ms=15,
                     mfc='none', mec=TRUE_C, mew=2.6)
        axes[1].plot([rec_row['pred_x'], rec_row['true_x']],
                     [rec_row['pred_y'], rec_row['true_y']], '-',
                     color='w', lw=1.2)
        err = rec_row['error_px']
        verdict = 'HIT' if err <= 5.0 else 'MISS'
        axes[1].set_title(f"{verdict} — error {err:.2f} px   "
                          f"(red = predicted, green = truth)", fontsize=11,
                          color=(TRUE_C if err <= 5.0 else PRED_C))

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Drift-Sense — {rec_row['id']}", fontsize=13)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def build_frames(data_dir, rows, hold):
    """(frame, seconds) pairs for the whole reel."""
    timeline = []
    for r in rows:
        ref_u8, search_u8 = E.read_pair(data_dir, r)
        for beat in (0, 1, 2):
            secs = hold * (2.0 if beat == 2 else 1.0)   # linger on the answer
            timeline.append((render_beat(ref_u8, search_u8, r, beat), secs))
        print(f"    {r['id']:10s} err={r['error_px']:8.2f}px  3 beats",
              flush=True)
    return timeline


def write_gif(timeline, path):
    imgs = [Image.fromarray(f) for f, _ in timeline]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=[int(s * 1000) for _, s in timeline], loop=0,
                 optimize=True)


def write_mp4(timeline, path, fps):
    import cv2
    h, w = timeline[0][0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    try:
        for frame, secs in timeline:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(max(int(round(secs * fps)), 1)):
                vw.write(bgr)
    finally:
        vw.release()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', default='data/train')
    p.add_argument('--out', default='results/demo.gif')
    p.add_argument('--n', type=int, default=5,
                   help='pairs in the reel (best n-1 plus the worst)')
    p.add_argument('--pairs', default=None,
                   help='comma-separated ids, overriding the auto selection')
    p.add_argument('--hold', type=float, default=1.2,
                   help='seconds per beat (the answer beat holds 2x)')
    p.add_argument('--mp4', action='store_true',
                   help='also write an .mp4 beside the gif')
    p.add_argument('--fps', type=int, default=12)
    args = p.parse_args(argv)

    records = E.load_pairs(args.data)
    if args.pairs:
        want = [s.strip() for s in args.pairs.split(',') if s.strip()]
        records = [r for r in records if r['id'] in want]
        if not records:
            sys.stderr.write(f"make_demo.py: no pair matched {args.pairs}\n")
            return 2

    print(f"Drift-Sense demo — running the pipeline over {len(records)} pairs")
    rows = E.evaluate_set(args.data, records, keep_maps=True)
    chosen = rows if args.pairs else pick_pairs(rows, args.n)
    print(f"  reel: {', '.join(r['id'] for r in chosen)}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    timeline = build_frames(args.data, chosen, args.hold)

    write_gif(timeline, args.out)
    total = sum(s for _, s in timeline)
    print(f"wrote {args.out} ({len(timeline)} frames, {total:.1f}s)")
    if args.mp4:
        mp4_path = os.path.splitext(args.out)[0] + '.mp4'
        write_mp4(timeline, mp4_path, args.fps)
        print(f"wrote {mp4_path} ({args.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
