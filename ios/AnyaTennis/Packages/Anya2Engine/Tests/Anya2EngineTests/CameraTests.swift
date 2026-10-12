// CameraTests.swift
//
// Camera registration (CameraRegistration, LiveCameraTrack) against camera.py.
//   - bookkeeping: the live track's committed warps equal the batch fill +
//     smooth on the same per-sample registrations (no frames needed)
//   - frames (ANYA2_CAMERA_CHECK, from tools/export_camera_check.py):
//       synthetic  camera.py's self-test: known jostles undone to < 1 px
//       real       a bumped stretch of Data/78: court corners within 1 px of
//                  where camera.py, run fresh on the same frames, puts them
//                  (main's CACHED track is one random draw of FLANN-LSH and
//                  differs from a re-run by 3-6 px; that is reported, not gated)
import Anya2Engine
@testable import Anya2Perception
import CoreGraphics
import Foundation
import ImageIO
import XCTest

final class CameraTests: XCTestCase {
    // ── bookkeeping ───────────────────────────────────────────────────────

    func testLiveTrackEqualsBatchFillAndSmooth() throws {
        let reference = try syntheticReference()
        var random = LiveAnalysisTests.SeededGenerator(state: 11)
        for trial in 0..<20 {
            let n = 1 + Int(random.next() % 40)
            // A few failures, sometimes leading, and a jump halfway.
            let warps: [Homography?] = (0..<n).map { i in
                if random.next() % 5 == 0 { return nil }
                let jump = i > n / 2 ? 9.0 : 0.0
                return Homography([1, 0, jump + Double(random.next() % 100) / 50, 0, 1, Double(random.next() % 100) / 50 - 1, 0, 0, 1])
            }
            let live = LiveCameraTrack(reference: reference, sampleStep: 6)
            var committedSoFar: [Homography] = []
            for (i, warp) in warps.enumerated() {
                live.add(warp: warp, sourceFrame: i * 6)
                // Committed warps never change once committed.
                XCTAssertEqual(Array(live.committed.prefix(committedSoFar.count)).map(\.m), committedSoFar.map(\.m))
                committedSoFar = live.committed
            }
            let liveTrack = live.finish()
            let batch = batchFillAndSmooth(warps, courtCorners: reference.courtCorners)
            XCTAssertEqual(liveTrack.warps.map(\.m), batch.map(\.m), "trial \(trial), n \(n)")
        }
    }

    /// camera._fill_failed_samples then camera.smooth, written out the batch way.
    private func batchFillAndSmooth(_ warps: [Homography?], courtCorners: [(x: Double, y: Double)]) -> [Homography] {
        let n = warps.count
        let firstGood = warps.firstIndex { $0 != nil }
        var filled: [Homography] = []
        var last: Homography?
        for w in warps {
            if let w { last = w }
            filled.append(last ?? .identity)
        }
        if let firstGood, firstGood > 0 { for i in 0..<firstGood { filled[i] = warps[firstGood]! } }
        let k = CameraRules.SMOOTH_SAMPLES
        if n < k { return filled }
        let points = analysisFrameCorners + courtCorners
        return (0..<n).map { i in
            let start = min(max(i - k / 2, 0), n - k)
            let located = (start..<(start + k)).map { j in points.map { filled[j].inverse.apply(x: $0.x, y: $0.y) } }
            var median: [Double] = []
            for p in 0..<points.count {
                median.append(Anya2Engine.median(located.map { $0[p].x }))
                median.append(Anya2Engine.median(located.map { $0[p].y }))
            }
            let target = points.flatMap { [$0.x, $0.y] }
            guard let m = Anya2OpenCVBridge.leastSquares(median, target) else { return filled[i] }
            return isSane(m) ? m : filled[i]
        }
    }

    // ── synthetic frames ──────────────────────────────────────────────────

    func testSyntheticJostlesAreUndone() throws {
        let dir = try checkDirectory("synthetic")
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("synthetic.json"))) as! [String: Any]
        let reference = try syntheticReference()
        let quad = (meta["court_quad"] as! [[Double]]).map { (x: $0[0], y: $0[1]) }
        for (name, rows) in meta["jostles"] as! [String: [[Double]]] {
            let a = Homography(rows.flatMap { $0 })
            let registration = try withGray(dir.appendingPathComponent("\(name).png")) { reference.register($0) }
            let warp = try XCTUnwrap(registration.warp, "\(name) did not register")
            let error = quad.map { corner -> Double in
                let moved = a.apply(x: corner.x, y: corner.y), back = warp.apply(x: moved.x, y: moved.y)
                return hypot(back.x - corner.x, back.y - corner.y)
            }.max()!
            print(String(format: "camera: %@ corner error %.2f px, %d inliers", name, error, registration.inliers))
            XCTAssertLessThan(error, 1.0, name)
        }
        let unmoved = try withGray(dir.appendingPathComponent("reference.png")) { reference.register($0) }
        let identityError = quad.map { c -> Double in
            let p = unmoved.warp!.apply(x: c.x, y: c.y)
            return max(abs(p.x - c.x), abs(p.y - c.y))
        }.max()!
        XCTAssertLessThan(identityError, 0.5, "unmoved frame -> identity")
        let noise = try withGray(dir.appendingPathComponent("noise.png")) { reference.register($0) }
        XCTAssertNil(noise.warp, "unrelated frame -> no warp")
        XCTAssertFalse(isSane(Homography([1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1])), "collapsed")
        XCTAssertFalse(isSane(Homography([-1, 0, 0, 0, 1, 0, 0, 0, 1])), "folded")
        XCTAssertTrue(isSane(.identity))
    }

    // ── a real bumped clip ────────────────────────────────────────────────

    func testRealClipMatchesMain() throws {
        let dir = try checkDirectory("real")
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("real.json"))) as! [String: Any]
        let corners = (meta["court_corners"] as! [[Double]]).map { (x: $0[0], y: $0[1]) }
        let reference = try withGray(dir.appendingPathComponent("reference.png")) {
            try CameraReference(calibrationFrame: $0, courtCorners: corners)
        }
        let samples = meta["samples"] as! [[String: Any]]
        let live = LiveCameraTrack(reference: reference, sampleStep: 1)
        for sample in samples {
            try withGray(dir.appendingPathComponent(sample["file"] as! String)) {
                live.add($0, sourceFrame: sample["frame"] as! Int)
            }
        }
        let track = live.finish()
        // The last two samples' smoothing windows reach past the exported
        // stretch in main's track, so only the others are comparable.
        func worstCornerDifference(_ key: String) -> Double {
            var worst = 0.0
            for (i, sample) in samples.enumerated().dropLast(2) {
                let python = Homography(sample[key] as! [Double])
                for c in corners {
                    let a = python.apply(x: c.x, y: c.y), b = track.warps[i].apply(x: c.x, y: c.y)
                    worst = max(worst, hypot(a.x - b.x, a.y - b.y))
                }
            }
            return worst
        }
        let fresh = worstCornerDifference("W_fresh"), cached = worstCornerDifference("W")
        print(String(format: "camera: Data/%@, %d samples, worst court-corner difference: %.3f px from camera.py "
                     + "run fresh, %.3f px from main's cached track", meta["clip"] as! String, samples.count - 2,
                     fresh, cached))
        XCTAssertLessThan(fresh, 1.0)
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private func checkDirectory(_ name: String) throws -> URL {
        guard let root = ProcessInfo.processInfo.environment["ANYA2_CAMERA_CHECK"] else {
            throw XCTSkip("set ANYA2_CAMERA_CHECK (tools/export_camera_check.py)")
        }
        return URL(fileURLWithPath: root).appendingPathComponent(name)
    }

    private func syntheticReference() throws -> CameraReference {
        // Without the exported frames, a flat grey frame with noise still
        // gives ORB something for the bookkeeping test (it never registers).
        if let root = ProcessInfo.processInfo.environment["ANYA2_CAMERA_CHECK"] {
            let dir = URL(fileURLWithPath: root).appendingPathComponent("synthetic")
            let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("synthetic.json"))) as! [String: Any]
            let quad = (meta["court_quad"] as! [[Double]]).map { (x: $0[0], y: $0[1]) }
            return try withGray(dir.appendingPathComponent("reference.png")) {
                try CameraReference(calibrationFrame: $0, courtCorners: quad)
            }
        }
        var random = LiveAnalysisTests.SeededGenerator(state: 3)
        var pixels = (0..<(960 * 540)).map { _ in UInt8(truncatingIfNeeded: random.next()) }
        return try pixels.withUnsafeMutableBufferPointer {
            try CameraReference(calibrationFrame: GrayFrame(pixels: UnsafePointer($0.baseAddress!), width: 960, height: 540, bytesPerRow: 960),
                                courtCorners: [(210, 470), (760, 470), (610, 180), (350, 180)])
        }
    }

    /// An 8-bit greyscale PNG's raw bytes, lent to `body`.
    private func withGray<T>(_ url: URL, _ body: (GrayFrame) throws -> T) throws -> T {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(image.bitsPerPixel, 8, "expected 8-bit greyscale: \(url.lastPathComponent)")
        let data = try XCTUnwrap(image.dataProvider?.data) as Data
        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            try body(GrayFrame(pixels: raw.bindMemory(to: UInt8.self).baseAddress!, width: image.width,
                               height: image.height, bytesPerRow: image.bytesPerRow))
        }
    }
}

/// The least-squares fit the live track uses, for the batch reimplementation above.
enum Anya2OpenCVBridge {
    static func leastSquares(_ source: [Double], _ target: [Double]) -> Homography? {
        smoothingFit(source: source, target: target)
    }
}
