#!/bin/bash
# build_opencv.sh -- the OpenCV the app links, for camera registration only
# (ORB, FLANN/brute-force matching, findHomography, fillConvexPoly, dilate).
#
# Builds OpenCV 4.12.0 with just core, imgproc, features2d, flann and calib3d
# for iPhone, the arm64 simulator and arm64 macOS (the SwiftPM tests), and
# puts opencv2.xcframework where Package.swift expects it:
#   Packages/Anya2Engine/Vendor/opencv2.xcframework   (git-ignored, ~60 MB)
# Takes ~30 min. Needs Xcode and cmake (brew install cmake).
#
#   ios/AnyaTennis/tools/build_opencv.sh [WORK_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../Packages/Anya2Engine/Vendor"
WORK="${1:-$(mktemp -d)}"
cd "$WORK"
test -d opencv || git clone -q --depth 1 -b 4.12.0 https://github.com/opencv/opencv.git
python3 opencv/platforms/apple/build_xcframework.py --out build \
    --iphoneos_archs arm64 --iphonesimulator_archs arm64 --macos_archs arm64 --build_only_specified_archs \
    --without=dnn --without=ml --without=objdetect --without=photo --without=stitching --without=video \
    --without=videoio --without=highgui --without=imgcodecs --without=gapi --without=objc --without=python \
    --without=java --without=js --without=ts --disable-bitcode -- -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# Repackage as static libraries + headers: the build's xcframework holds
# static .frameworks, which Xcode would wrongly embed in the app (the
# simulator then refuses to install it).
rm -rf static && args=()
for slice in ios-arm64 ios-arm64-simulator macos-arm64; do
    framework="build/opencv2.xcframework/$slice/opencv2.framework"
    mkdir -p "static/$slice/include"
    cp -RL "$framework/Headers" "static/$slice/include/opencv2"
    cp "$framework/opencv2" "static/$slice/libopencv2.a"
    args+=(-library "static/$slice/libopencv2.a" -headers "static/$slice/include")
done
mkdir -p "$DEST"
rm -rf "$DEST/opencv2.xcframework"
xcodebuild -create-xcframework "${args[@]}" -output "$DEST/opencv2.xcframework"
echo "installed $DEST/opencv2.xcframework"
