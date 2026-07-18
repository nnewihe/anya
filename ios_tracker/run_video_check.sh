#!/bin/zsh
# Compile and run the macOS end-to-end video check against the real app sources.
set -e
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
OUT="${TMPDIR:-/tmp}/balltracker_video"
VIDEO="${1:-/Users/tennis/Documents/Code/Laptop/src/anya/archive/out.mp4}"

swiftc -O -parse-as-library -o "$OUT" \
    ParityCheck/video_main.swift \
    BallTracker/Detection/Letterbox.swift \
    BallTracker/Detection/BallDetector.swift \
    BallTracker/Detection/ExclusionZones.swift \
    BallTracker/Detection/PlayerDetector.swift \
    BallTracker/Tracking/Matrix.swift \
    BallTracker/Tracking/KalmanIMM.swift \
    BallTracker/Tracking/BallTrackManager.swift \
    BallTracker/Tracking/RallyDetector.swift \
    BallTracker/Tracking/TrackerEngine.swift \
    BallTracker/Video/CheckpointStore.swift \
    BallTracker/Video/VideoProcessor.swift

ANYA_REPO="$REPO" VIDEO="$VIDEO" DUMP_CSV="${DUMP_CSV:-}" "$OUT"
