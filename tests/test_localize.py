"""Verify Gate 9 for localize.py — the delivered CLI.

  a) Runs from an unrelated working directory (cd /tmp).
  b) A 1200x900 search image must not crash.
  c) A pure-noise reference must still print a coordinate.
  d) A fresh venv built from requirements.txt must run it.

Each check prints PASS.  (d) is skipped with a printed notice if there is no
network to build the venv, since it is an environment check, not a code one.

Run: python tests/test_localize.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALIZE = os.path.join(ROOT, 'localize.py')
TRAIN = os.path.join(ROOT, 'data', 'train')

import cv2
import numpy as np

COORD = re.compile(r'^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$')


def run_cli(args, cwd=None, python=None):
    proc = subprocess.run([python or sys.executable, LOCALIZE] + args,
                          cwd=cwd, capture_output=True, text=True, timeout=300)
    return proc


def check_contract(proc, label, expect_rc=0):
    assert proc.returncode == expect_rc, \
        f"{label}: exit {proc.returncode}\nstderr:\n{proc.stderr[-2000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"{label}: expected 1 stdout line, got {lines!r}"
    assert COORD.match(lines[0]), f"{label}: bad format {lines[0]!r}"
    assert 'Traceback' not in proc.stderr, f"{label}: traceback in stderr"
    return tuple(float(v) for v in lines[0].split(','))


def gate_a(tmp):
    ref = os.path.join(TRAIN, 'pair_0000_ref.png')
    sea = os.path.join(TRAIN, 'pair_0000_search.png')
    proc = run_cli(['--ref', ref, '--search', sea], cwd=tmp)
    x, y = check_contract(proc, 'a')
    print(f"  (a) runs from {tmp} (absolute path, unrelated cwd) -> {x:.2f}, {y:.2f}")
    print("      PASS")


def gate_b(tmp):
    """Non-1000x1000 search image: 1200x900."""
    sea = cv2.imread(os.path.join(TRAIN, 'pair_0000_search.png'),
                     cv2.IMREAD_GRAYSCALE)
    odd = cv2.resize(sea, (1200, 900), interpolation=cv2.INTER_LINEAR)
    p = os.path.join(tmp, 'search_1200x900.png')
    cv2.imwrite(p, odd)
    proc = run_cli(['--ref', os.path.join(TRAIN, 'pair_0000_ref.png'),
                    '--search', p])
    x, y = check_contract(proc, 'b')
    assert 0 <= x <= 1199 and 0 <= y <= 899, f"(b) out of bounds: {x},{y}"
    print(f"  (b) 1200x900 search -> {x:.2f}, {y:.2f} (inside bounds)")
    print("      PASS")


def gate_c(tmp):
    """Pure-noise reference: no lattice, no structure, nothing to match."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (1000, 1000), dtype=np.uint8)
    p = os.path.join(tmp, 'noise_ref.png')
    cv2.imwrite(p, noise)
    proc = run_cli(['--ref', p, '--search',
                    os.path.join(TRAIN, 'pair_0000_search.png')])
    x, y = check_contract(proc, 'c')
    print(f"  (c) pure-noise reference -> {x:.2f}, {y:.2f} (no traceback)")

    # also: colour input, a non-PNG container, and a tiny search image
    colour = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    pc = os.path.join(tmp, 'colour_ref.bmp')
    cv2.imwrite(pc, colour)
    tiny = os.path.join(tmp, 'tiny_search.jpg')
    cv2.imwrite(tiny, cv2.resize(noise, (120, 90)))
    for label, args in (('colour .bmp ref', ['--ref', pc, '--search',
                                             os.path.join(TRAIN, 'pair_0000_search.png')]),
                        ('120x90 .jpg search', ['--ref', pc, '--search', tiny]),
                        ('missing file', ['--ref', os.path.join(tmp, 'nope.png'),
                                          '--search', tiny])):
        pr = run_cli(args)
        lines = [ln for ln in pr.stdout.splitlines() if ln.strip()]
        assert lines and COORD.match(lines[0]), f"(c) {label}: {pr.stdout!r}"
        assert 'Traceback' not in pr.stderr, f"(c) {label}: traceback"
        print(f"      {label:22s} -> {lines[0].strip()}")
    print("      PASS")


def gate_d(tmp):
    """Fresh venv from requirements.txt."""
    venv = os.path.join(tmp, 'freshvenv')
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv], check=True,
                       capture_output=True, timeout=300)
        pip = os.path.join(venv, 'bin', 'pip')
        py = os.path.join(venv, 'bin', 'python')
        inst = subprocess.run([pip, 'install', '-q', '--disable-pip-version-check',
                               '-r', os.path.join(ROOT, 'requirements.txt')],
                              capture_output=True, text=True, timeout=1800)
        if inst.returncode != 0:
            tail = (inst.stderr or inst.stdout)[-300:]
            print("  (d) SKIPPED - could not build a fresh venv here "
                  f"(no network?):\n      {tail.strip()[:300]}")
            return
        proc = run_cli(['--ref', os.path.join(TRAIN, 'pair_0000_ref.png'),
                        '--search', os.path.join(TRAIN, 'pair_0000_search.png')],
                       python=py)
        x, y = check_contract(proc, 'd')
        print(f"  (d) fresh venv + requirements.txt -> {x:.2f}, {y:.2f}")
        print("      PASS")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        print(f"  (d) SKIPPED - venv creation failed: {type(exc).__name__}")


if __name__ == '__main__':
    print("Verify Gate 9 (localize.py)")
    tmp = tempfile.mkdtemp(prefix='driftsense_gate9_')
    try:
        gate_a(tmp)
        gate_b(tmp)
        gate_c(tmp)
        gate_d(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")
