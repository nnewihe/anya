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
        let directories = try FileManager.default
            .contentsOfDirectory(at: URL(fileURLWithPath: root), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("meta.json").path) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return try directories.map(ReplayClip.init(directory:))
    }

    func testTracksMatchPython() throws {
        for clip in try corpusClips() {
            let tracks = buildTracks(near: clip.poseDetections("near"), far: clip.poseDetections("far"),
                                     geometry: clip.geometry)
            assertSame(tracks.boxes, clip.referenceFloat32("tracks.boxes"), "\(clip.meta.clip) boxes")
            assertSame(tracks.keypoints, clip.referenceFloat32("tracks.keypoints"), "\(clip.meta.clip) keypoints")
            assertSame(tracks.court, clip.referenceFloat32("tracks.court"), "\(clip.meta.clip) court", tolerance: 1e-4)
            let wantEligible = clip.referenceBool("tracks.eligible")
            let eligibleDiffs = zip(tracks.eligible, wantEligible).enumerated().filter { $0.element.0 != $0.element.1 }
            XCTAssertEqual(eligibleDiffs.count, 0, "\(clip.meta.clip) eligible: \(eligibleDiffs.count) of "
                           + "\(wantEligible.count) differ, first at \(eligibleDiffs.prefix(5).map(\.offset))")
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
