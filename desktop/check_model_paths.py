"""
check_model_paths.py — fail the build if a model default would auto-download.

Run by build_macos.sh before PyInstaller.

The bug this exists to catch: handed a bare name like "yolov8n-pose.pt",
ultralytics looks in the current working directory and then DOWNLOADS the
weights from the internet. Every default on the reel's path must therefore be
an absolute path to a real file, or the packaged app ignores its own bundled
copy and fetches one at the start of every job.

That shipped in 0.1.0-beta.5 and was invisible in testing because the download
simply succeeded for anyone online — it took a tester on a machine that
couldn't reach the internet to surface it, as a download failure mid-run. The
failure mode is silent-until-it-isn't, so it needs a build-time check rather
than a code review habit.

Only the defaults actually reachable from the desktop app are checked. Research
and CLI modules under pipeline/ still use bare names deliberately; they are not
bundled and are run by someone who can see the download happen.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    failures = []

    def check(label, value):
        if not value:
            failures.append(f"{label}: empty")
        elif not os.path.isabs(str(value)):
            failures.append(
                f"{label}: {value!r} is not an absolute path — ultralytics will "
                f"download this instead of using the bundled copy"
            )
        elif not os.path.isfile(str(value)):
            failures.append(f"{label}: {value} does not exist")
        else:
            print(f"  ok  {label}\n      {value}")

    from walking.extract_pose import DEFAULT_POSE_MODEL
    check("walking.extract_pose.DEFAULT_POSE_MODEL", DEFAULT_POSE_MODEL)

    from pipeline.anya_end_telemetry import EndExtractorConfig
    check("EndExtractorConfig.pose_model", EndExtractorConfig().pose_model)

    from pipeline.utilities import Config
    check("Config.DEFAULT_NEAR_TROPHY_MODEL_PATH",
          Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

    # The weights the spec bundles, at the paths the pipeline resolves them by.
    repo = Path(__file__).resolve().parent.parent
    for rel in ("pipeline/models/ball_best.pt",
                "pipeline/models/yolo26n.pt",
                "pipeline/models/yolov8n-pose.pt",
                "pipeline/models/trophy_best.pt",
                "walking/outputs/walking_model.joblib",
                "walking/outputs/walking_model_15hz.joblib"):
        check(f"bundled weight {rel}", repo / rel)

    if failures:
        print("\nerror: model paths would trigger a runtime download:",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("  all model defaults resolve to bundled files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
