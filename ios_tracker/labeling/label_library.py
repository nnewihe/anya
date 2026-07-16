#!/usr/bin/env python3
"""Label the rally library, clip by clip.

    python3 label_library.py --lib library            # next unlabelled clip
    python3 label_library.py --lib library --all      # keep going until done
    python3 label_library.py --lib library --status   # progress only
    python3 label_library.py --lib library --clip 22_0

Each clip opens in the same labeller as before: numbered candidates from the
real detector, press a digit to accept one, `n` for no ball. Fully resumable —
stop any time, progress is saved per clip.

`step` comes from the manifest so a 120 fps clip costs the same keystrokes as a
30 fps one for the same tennis; the scorer ignores frames you never labelled.

Label the no-ball frames too. Drawing a ball where there is none is the
tracker's real failure mode, and it is invisible unless the library says
"nothing here".
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def labels_path(lib, name):
    return os.path.join(lib, "labels", name + ".json")


def clip_status(lib, c):
    p = labels_path(lib, c["name"])
    if not os.path.exists(p):
        return 0, 0, 0
    d = json.load(open(p))["labels"]
    ball = sum(1 for v in d.values() if v)
    none = sum(1 for v in d.values() if v is None)
    return len(d), ball, none


def expected_labels(c):
    frames = int(c["dur"] * c["fps"])
    return max(1, frames // max(c["step"], 1))


def show_status(lib, man):
    print(f"{'clip':<10} {'match':<6} {'labelled':>9} {'ball':>5} {'none':>5} {'target':>7}  {'':<4}")
    tot = tb = tn = texp = 0
    for c in man["clips"]:
        n, b, nn = clip_status(lib, c)
        exp = expected_labels(c)
        tot += n; tb += b; tn += nn; texp += exp
        mark = "done" if n >= exp * 0.9 else ("part" if n else "")
        print(f"{c['name']:<10} {c['match']:<6} {n:>9} {b:>5} {nn:>5} {exp:>7}  {mark:<4}")
    print(f"\n{tot}/{texp} labels ({tb} ball, {tn} no-ball) across {len(man['clips'])} clips "
          f"from {len({c['match'] for c in man['clips']})} matches")
    return tot, texp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default="library")
    ap.add_argument("--all", action="store_true", help="continue to the next clip after each")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clip", help="label this clip by name")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.lib, "manifest.json")))
    os.makedirs(os.path.join(a.lib, "labels"), exist_ok=True)

    if a.status:
        show_status(a.lib, man)
        return

    todo = man["clips"]
    if a.clip:
        todo = [c for c in todo if c["name"] == a.clip]
        if not todo:
            sys.exit(f"no clip named {a.clip}")
    else:
        todo = [c for c in todo
                if clip_status(a.lib, c)[0] < expected_labels(c) * 0.9]
        if not todo:
            print("everything is labelled.")
            show_status(a.lib, man)
            return
        if not a.all:
            todo = todo[:1]

    for i, c in enumerate(todo, 1):
        n, b, nn = clip_status(a.lib, c)
        print(f"\n=== [{i}/{len(todo)}] {c['name']}  (match {c['match']}, "
              f"{c['dur']}s @{c['fps']}fps, {c['w']}x{c['h']}, step {c['step']}) "
              f"— {n} labels so far ===")
        r = subprocess.run([sys.executable, os.path.join(HERE, "label_ball.py"),
                            "--video", c["clip"],
                            "--out", labels_path(a.lib, c["name"]),
                            "--step", str(c["step"])])
        if r.returncode != 0:
            print("labeller exited non-zero; stopping")
            break

    print()
    show_status(a.lib, man)


if __name__ == "__main__":
    main()
