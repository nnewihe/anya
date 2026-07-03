"""
Python oracle run on the user's real match clip (langmead_1min.mov), using the
REAL cached court corners / exclusion zones / active zone the user's own
`python3 -m pipeline.rally_detector` run produced next to the video. No
monkeypatching -- this is the unmodified pipeline with unmodified inputs, so
its output should exactly reproduce what the user saw (validates this script
before using it for iOS-vs-Python parity comparisons with the app's own
tapped corners).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

CLIP = Path("/Users/tennis/Documents/match_play/data/langmead_1min.mov")

from rally_detector import collect_rally_segments  # noqa: E402


def main():
    if not CLIP.exists():
        raise SystemExit(f"clip not found: {CLIP}")
    segs = collect_rally_segments(str(CLIP), headless=True, start_frame=0)
    out = [
        {"start": round(float(s), 2), "end": round(float(e), 2), "origin": o}
        for s, e, o in segs
    ]
    result = {"clip": str(CLIP), "segment_count": len(out), "segments": out}
    dst = REPO / "spikes" / "fixtures" / "langmead" / "oracle_result_real_corners.json"
    dst.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
