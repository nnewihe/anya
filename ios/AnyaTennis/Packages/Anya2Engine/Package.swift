// swift-tools-version:5.10
//
// Anya2Engine: the anya2 point-detection pipeline (pipeline/anya2 on main),
// ported to Swift. Pure computation -- no camera, no UI -- so it builds and
// tests on macOS as well as iOS, and the app target links it.
import PackageDescription

let package = Package(
    name: "Anya2Engine",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Anya2Engine", targets: ["Anya2Engine"]),
        .library(name: "Anya2Perception", targets: ["Anya2Perception"]),
        .executable(name: "anya2-replay", targets: ["Anya2Replay"]),
    ],
    targets: [
        .target(name: "Anya2Engine",
                // walking_model.json: the walking classifier's trees
                // (tools/export_walking_model.py).
                resources: [.copy("Resources/walking_model.json")]),
        // Frames -> people: the Ultralytics letterbox, the Core ML pose model
        // and its decode, the two passes (perceive.py). Separate so the
        // engine itself stays free of Core ML and Core Video.
        .target(name: "Anya2Perception", dependencies: ["Anya2Engine", "Anya2OpenCV"]),
        // The OpenCV calls camera registration makes, behind an Objective-C++
        // shim so no C++ reaches Swift. Vendor/opencv2.xcframework is built
        // by tools/build_opencv.sh (not checked in: ~60 MB).
        .target(name: "Anya2OpenCV", dependencies: ["opencv2"],
                linkerSettings: [.linkedLibrary("c++"), .linkedFramework("Accelerate"),
                                 .linkedFramework("OpenCL", .when(platforms: [.macOS]))]),
        .binaryTarget(name: "opencv2", path: "Vendor/opencv2.xcframework"),
        // Replays Python's cached pose detections through the Swift pipeline,
        // for the corpus parity check (tools/export_replay.py writes the input).
        .executableTarget(name: "Anya2Replay", dependencies: ["Anya2Engine"]),
        .testTarget(name: "Anya2EngineTests", dependencies: ["Anya2Engine", "Anya2Perception"]),
    ]
)
