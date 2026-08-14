#!/bin/zsh
# Compile and run the JVM tracker-parity harness against the REAL app sources.
#
# Mirrors ios_tracker/run_parity_check.sh, minus the on-device detection parity
# (that needs the TFLite model on an accelerator; run it on a device/emulator
# with `./gradlew :app:connectedDebugAndroidTest` once instrumentation exists).
#
# What this covers: the 10-scenario tracker oracle (same scenarios as
# pipeline/ball_tracker.py / mobile/test/ball_tracker_test.dart) plus the DBSCAN
# exclusion-zone clustering and the offline Viterbi solver — all the pure
# Kotlin ports, exercising the shipped Kalman/IMM/BallTrackManager code.
set -e
cd "$(dirname "$0")"

# Prefer Android Studio's bundled JDK if JAVA_HOME isn't already a JDK 17+.
if [ -z "$JAVA_HOME" ] && [ -d "/Applications/Android Studio.app/Contents/jbr/Contents/Home" ]; then
    export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
fi

./gradlew :app:testDebugUnitTest "$@"
echo
echo "Report: app/build/reports/tests/testDebugUnitTest/index.html"
