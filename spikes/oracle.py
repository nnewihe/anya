"""
Python oracle for engine parity: run the real pipeline's collect_rally_segments
on the same clip and corners the Dart engine uses, and dump the segments.

We monkeypatch init_court to return fixed corners (no interactive selection) and
force a full-frame active zone, matching the Dart engine's configuration.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

CLIP = REPO / "spikes" / "fixtures" / "clip" / "clip30.mp4"
# Same order as the Dart harness: [BL, BR, TR, TL] in 960x540 space.
CORNERS = [[120, 510], [840, 510], [600, 140], [360, 140]]

import utilities  # noqa: E402


def _fake_init_court(video_path, target_idx=300, analysis_size=None):
    return [tuple(p) for p in CORNERS], (540, 960, 3)


utilities.init_court = _fake_init_court

# Full-frame active zone + skip exclusion scan for an apples-to-apples run.
import anya_base  # noqa: E402

anya_base.init_court = _fake_init_court
_orig_create = utilities.create_auto_exclusion_zones
utilities.create_auto_exclusion_zones = lambda *a, **k: []
anya_base.create_auto_exclusion_zones = lambda *a, **k: []
utilities.load_cached_exclusion_zones = lambda *a, **k: []

# Write a full-frame active zone cache next to the clip.
W, H = 960, 540
zone = [[0, 0], [W // 2, 0], [W, 0], [W, H // 2], [W, H], [W // 2, H], [0, H], [0, H // 2]]
(CLIP.parent / "active_zone_config.json").write_text(json.dumps(zone))

from rally_detector import collect_rally_segments  # noqa: E402


def main():
    segs = collect_rally_segments(str(CLIP), headless=True, start_frame=0)
    out = [
        {"start": round(float(s), 2), "end": round(float(e), 2), "origin": o}
        for s, e, o in segs
    ]
    result = {"clip": str(CLIP), "segment_count": len(out), "segments": out}
    dst = REPO / "spikes" / "fixtures" / "oracle_clip30.json"
    dst.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
