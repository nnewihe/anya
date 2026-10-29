// LiveMatchAnalysisTests.swift
//
// The join of the pose clock and the camera clock (LiveMatchAnalysis): fed
// one sample at a time, with camera registrations arriving every few
// samples, its result must equal the batch pipeline run on the same
// detections with the same final camera track. Uses each corpus clip's
// own camera track (main's warps) as the registrations.
import Anya2Engine
@testable import Anya2Perception
import XCTest

final class LiveMatchAnalysisTests: XCTestCase {
    func testJoinedLiveEqualsBatch() throws {
        let model = try WalkingModel.bundled()
        var random = LiveAnalysisTests.SeededGenerator(state: 5)
        var noise = (0..<(960 * 540)).map { _ in UInt8(truncatingIfNeeded: random.next()) }
        for clip in try CorpusParityTests().corpusClips() {
            let near = clip.poseDetections("near"), far = clip.poseDetections("far")
            let corners = clip.meta.corners.map { (x: $0[0], y: $0[1]) }
            let reference = try noise.withUnsafeMutableBufferPointer {
                try CameraReference(calibrationFrame: GrayFrame(pixels: UnsafePointer($0.baseAddress!), width: 960,
                                                                height: 540, bytesPerRow: 960), courtCorners: corners)
            }
            let sourceFPS = near.fps * Double(near.stride)
            let live = LiveMatchAnalysis(reference: reference, referenceHomography: Homography(courtCorners: corners),
                                         sourceFPS: sourceFPS, walkingModel: model)
            XCTAssertEqual(live.poseStride, near.stride)
            let pythonTrack = clip.geometry.cameraTrack
            let total = min(near.samples, far.samples)
            for i in 0..<total {
                let frame = i * near.stride
                var rows = [PassRow(), PassRow()]
                for (pass, detections) in [near, far].enumerated() {
                    let people = min(detections.people, PerceiveRules.MAX_PERSONS)
                    for p in 0..<people {
                        let k = (i * detections.people + p)
                        rows[pass].confidences[p] = detections.confidences[k]
                        for j in 0..<4 { rows[pass].boxes[p * 4 + j] = detections.boxes[k * 4 + j] }
                        for j in 0..<51 { rows[pass].keypoints[p * 51 + j] = detections.keypoints[k * 51 + j] }
                    }
                }
                let warp: Homography?? = live.camera.isSampleFrame(frame)
                    ? .some(pythonTrack.warps[pythonTrack.sampleIndex(nearestTo: frame)]) : nil
                live.replay(PerceivedSample(near: rows[0], far: rows[1]), sourceFrame: frame, cameraWarp: warp)
            }
            let result = live.finish()
            let batch = runBatch(near: near, far: far,
                                 geometry: Geometry(referenceHomography: Homography(courtCorners: corners),
                                                    cameraTrack: live.geometry.cameraTrack),
                                 walkingModel: model)
            let name = clip.meta.clip
            XCTAssertEqual(result.live, batch.live, "\(name): live score")
            for (a, b) in [(result.nearServes, batch.nearServes), (result.farServes, batch.farServes),
                           (result.pointEnds, batch.pointEnds)] {
                XCTAssertEqual(a.map(\.t), b.map(\.t), "\(name): events")
            }
            XCTAssertEqual(result.reel.segments.map(\.start), batch.reel.segments.map(\.start), "\(name): segments")
            XCTAssertEqual(result.reel.segments.map(\.stop), batch.reel.segments.map(\.stop), "\(name): segments")
            XCTAssertTrue(live.finish().reel.segments.count == result.reel.segments.count, "finish is repeatable")
        }
    }
}
