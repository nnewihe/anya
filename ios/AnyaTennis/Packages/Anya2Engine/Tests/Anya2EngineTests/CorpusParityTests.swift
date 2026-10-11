// Parity against main's Python pipeline on the real corpus clips.
//
// Needs the replay export (ios/AnyaTennis/tools/export_replay.py). Point
// ANYA2_REPLAY at its output directory; without it these tests are skipped,
// so the package's own unit tests still run anywhere.
import XCTest
@testable import Anya2Engine

final class CorpusParityTests: XCTestCase {
    /// Every exported clip, or a skip when no export is configured.
    func corpusClips() throws -> [ReplayClip] {
        guard let root = ProcessInfo.processInfo.environment["ANYA2_REPLAY"] else {
            throw XCTSkip("set ANYA2_REPLAY to an export_replay.py output directory")
        }
        // ANYA2_CLIPS="22 25" limits the run to those clips while debugging.
        let only = Set((ProcessInfo.processInfo.environment["ANYA2_CLIPS"] ?? "").split(separator: " ").map(String.init))
        let directories = try FileManager.default
            .contentsOfDirectory(at: URL(fileURLWithPath: root), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("meta.json").path) }
            .filter { only.isEmpty || only.contains($0.lastPathComponent) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return try directories.map(ReplayClip.init(directory:))
    }

    func testHomographyMatchesOpenCV() throws {
        for clip in try corpusClips() {
            guard let python = clip.meta.homography else { continue }
            // To 1e-12 relative, not bit-exact: OpenCV's compiled LU fuses some
            // multiply-adds, which moves the last digit or two. Tracks, which
            // read this map, still match Python exactly (testTracksMatchPython).
            assertSame(clip.geometry.referenceHomography.m, python, "\(clip.meta.clip) homography",
                       tolerance: 1e-12)
        }
    }

    func testTracksMatchPython() throws {
        for clip in try corpusClips() {
            let tracks = buildTracks(near: clip.poseDetections("near"), far: clip.poseDetections("far"),
                                     geometry: clip.geometry)
            assertSame(tracks.boxes, clip.referenceFloat32("tracks.boxes"), "\(clip.meta.clip) boxes")
            assertSame(tracks.keypoints, clip.referenceFloat32("tracks.keypoints"), "\(clip.meta.clip) keypoints")
            assertSame(tracks.court, clip.referenceFloat32("tracks.court"), "\(clip.meta.clip) court")
            let wantEligible = clip.referenceBool("tracks.eligible")
            let eligibleDiffs = zip(tracks.eligible, wantEligible).enumerated().filter { $0.element.0 != $0.element.1 }
            XCTAssertEqual(eligibleDiffs.count, 0, "\(clip.meta.clip) eligible: \(eligibleDiffs.count) of "
                           + "\(wantEligible.count) differ, first at \(eligibleDiffs.prefix(5).map(\.offset))")
        }
    }
}

extension CorpusParityTests {
    func testEndSignalsMatchPython() throws {
        let model = try WalkingModel.bundled()
        for clip in try corpusClips() {
            let tracks = buildTracks(near: clip.poseDetections("near"), far: clip.poseDetections("far"),
                                     geometry: clip.geometry)
            let signals = computeEndSignals(tracks: tracks, fixedHomography: clip.geometry.referenceHomography,
                                            walkingModel: model)
            assertSame(signals.walkingProbability, clip.referenceFloat64("walking.probability"),
                       "\(clip.meta.clip) walking probability", tolerance: 1e-6)
            for name in NearEndRules.SIGNAL_NAMES {
                assertSame(signals.nearEnd[name]!, clip.referenceFloat32("near_end.\(name)").map(Double.init),
                           "\(clip.meta.clip) near_end \(name)", tolerance: 1e-6)
            }
        }
    }
}

/// Element-wise equality with NaN == NaN; reports the first few mismatches.
func assertSame<T: BinaryFloatingPoint>(_ got: [T], _ want: [T], _ label: String,
                                        tolerance: Double = 0, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(got.count, want.count, "\(label): length", file: file, line: line)
    var mismatches: [String] = []
    var count = 0
    for i in 0..<min(got.count, want.count) {
        let g = Double(got[i]), w = Double(want[i])
        let same = (g.isNaN && w.isNaN) || g == w || abs(g - w) <= tolerance * (1 + abs(w))
        if !same {
            count += 1
            if mismatches.count < 6 { mismatches.append("[\(i)] got \(g) want \(w)") }
        }
    }
    XCTAssertEqual(count, 0, "\(label): \(count) mismatches\n" + mismatches.joined(separator: "\n"),
                   file: file, line: line)
}
