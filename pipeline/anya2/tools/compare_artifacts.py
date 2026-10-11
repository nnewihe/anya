"""
compare_artifacts.py
====================
Are two anya2 runs identical?  Compares every .npz and .json file that two
artifact directories have in common, and reports the first difference in each.

Use it whenever you change the pipeline and expect the output NOT to change
(a refactor, a rename, a speed-up):

    python -m pipeline.anya2.tools.compare_artifacts OLD_DIR NEW_DIR

  * .npz  -- every array must have the same keys, dtype and shape, and equal
             values (NaN counts as equal to NaN).
  * .json -- the parsed content must be equal, and so must the raw bytes.

Exit code 0 when everything matches, 1 otherwise.
"""

import argparse
import json
import os
import sys

import numpy as np

# Files that are inputs copied into a run directory, not outputs of it.
DEFAULT_SKIP = ("_court_cache.json",)


def compare_npz(path_a, path_b):
    """None if identical, else a one-line description of the first difference."""
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return f"keys differ: {sorted(a.files)} vs {sorted(b.files)}"
        for key in sorted(a.files):
            x, y = a[key], b[key]
            if x.dtype != y.dtype:
                return f"{key}: dtype {x.dtype} vs {y.dtype}"
            if x.shape != y.shape:
                return f"{key}: shape {x.shape} vs {y.shape}"
            nan_ok = x.dtype.kind in "fc"
            if not np.array_equal(x, y, equal_nan=nan_ok):
                bad = np.argwhere(~((x == y) | (np.isnan(x) & np.isnan(y))
                                    if nan_ok else (x == y)))
                where = tuple(int(i) for i in bad[0]) if len(bad) else ()
                return (f"{key}: {len(bad)} values differ, first at {where}: "
                        f"{x[where]!r} vs {y[where]!r}")
    return None


def compare_json(path_a, path_b):
    with open(path_a, "rb") as fh:
        raw_a = fh.read()
    with open(path_b, "rb") as fh:
        raw_b = fh.read()
    if raw_a == raw_b:
        return None
    if json.loads(raw_a) == json.loads(raw_b):
        return "same content, different bytes (formatting or key order)"
    return _first_json_difference(json.loads(raw_a), json.loads(raw_b), "")


def _first_json_difference(a, b, where):
    if type(a) is not type(b):
        return f"{where or '<root>'}: {a!r} vs {b!r}"
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{where or '<root>'}: keys {list(a)} vs {list(b)}"
        for key in a:
            found = _first_json_difference(a[key], b[key], f"{where}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{where or '<root>'}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_json_difference(x, y, f"{where}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{where or '<root>'}: {a!r} vs {b!r}"


def compare_dirs(dir_a, dir_b, skip=DEFAULT_SKIP, verbose=True):
    """Compare every shared .npz/.json. Returns {filename: difference-or-None}."""
    names_a = {f for f in os.listdir(dir_a) if f.endswith((".npz", ".json"))}
    names_b = {f for f in os.listdir(dir_b) if f.endswith((".npz", ".json"))}
    results = {}
    for name in sorted(names_a | names_b):
        if name.endswith(skip):
            continue
        if name not in names_a or name not in names_b:
            results[name] = f"only in {'A' if name in names_a else 'B'}"
            continue
        compare = compare_npz if name.endswith(".npz") else compare_json
        results[name] = compare(os.path.join(dir_a, name), os.path.join(dir_b, name))
    if verbose:
        for name, diff in results.items():
            print(f"  {'SAME' if diff is None else 'DIFF'}  {name}"
                  + ("" if diff is None else f"  -- {diff}"))
    return results


def main():
    ap = argparse.ArgumentParser(description="Compare two anya2 artifact dirs")
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    args = ap.parse_args()
    results = compare_dirs(args.dir_a, args.dir_b)
    n_diff = sum(d is not None for d in results.values())
    print(f"{len(results) - n_diff}/{len(results)} files identical")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
