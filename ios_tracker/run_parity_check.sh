#!/bin/zsh
# Compile and run the macOS parity harness against the real app sources.
set -e
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
OUT="${TMPDIR:-/tmp}/balltracker_parity"

swiftc -O -o "$OUT" \
    ParityCheck/main.swift \
    BallTracker/Detection/Letterbox.swift \
    BallTracker/Detection/BallDetector.swift \
    BallTracker/Tracking/Matrix.swift \
    BallTracker/Tracking/KalmanIMM.swift \
    BallTracker/Tracking/BallTrackManager.swift

ANYA_REPO="$REPO" "$OUT"
