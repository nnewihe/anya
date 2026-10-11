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
        .executable(name: "anya2-replay", targets: ["Anya2Replay"]),
    ],
    targets: [
        .target(name: "Anya2Engine",
                // walking_model.json: the walking classifier's trees
                // (tools/export_walking_model.py).
                resources: [.copy("Resources/walking_model.json")]),
        // Replays Python's cached pose detections through the Swift pipeline,
        // for the corpus parity check (tools/export_replay.py writes the input).
        .executableTarget(name: "Anya2Replay", dependencies: ["Anya2Engine"]),
        .testTarget(name: "Anya2EngineTests", dependencies: ["Anya2Engine"]),
    ]
)
