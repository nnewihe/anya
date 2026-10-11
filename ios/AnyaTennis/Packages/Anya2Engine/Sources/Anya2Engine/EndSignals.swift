// EndSignals.swift  --  port of run._compute_end_signals (stage 6)
//
// The walking probability and near_end's four signals, both read off ONE
// near-player pose track: the near slot with more coverage leads, and
// wherever it is empty the other near slot fills in (on a changeover the same
// person can move slots).
import Foundation

public struct EndSignals {
    public let fps: Double
    public let walkingProbability: [Double]
    /// near_end's four signals, rounded to float32 as Python stores them.
    public let nearEnd: [String: [Double]]
}

/// The near-player track both end signals read (run._near_player_pose_for_end_signals).
/// `onCourt` is the LEAD slot's eligibility even where the other slot filled
/// in -- kept as Python has it.
public func nearPlayerPoseForEndSignals(_ tracks: PlayerTracks)
    -> (keypoints: SampleTable, boxes: SampleTable, onCourt: [Double]) {
    let coverage = Slots.near.map { slot in
        Double((0..<tracks.samples).filter { tracks.boxes[($0 * Slots.count + slot) * 4].isFinite }.count)
            / Double(tracks.samples)
    }
    let (lead, other) = coverage[0] >= coverage[1] ? (0, 1) : (1, 0)
    var keypoints = tracks.keypoints(slot: lead), boxes = tracks.boxes(slot: lead)
    let otherKeypoints = tracks.keypoints(slot: other), otherBoxes = tracks.boxes(slot: other)
    for i in 0..<tracks.samples where !boxes[i, 0].isFinite && otherBoxes[i, 0].isFinite {
        for c in 0..<51 { keypoints[i, c] = otherKeypoints[i, c] }
        for c in 0..<4 { boxes[i, c] = otherBoxes[i, c] }
    }
    let onCourt = tracks.eligible(slot: lead).map { $0 ? 1.0 : 0.0 }
    return (keypoints, boxes, onCourt)
}

/// Stage 6. `fixedHomography` is the calibration homography alone: the
/// walking classifier never used the camera track.
public func computeEndSignals(tracks: PlayerTracks, fixedHomography: Homography,
                              walkingModel: WalkingModel) -> EndSignals {
    let pose = nearPlayerPoseForEndSignals(tracks)
    let frameSignals = walkingFrameSignals(keypoints: pose.keypoints, boxes: pose.boxes,
                                           homography: fixedHomography, fps: tracks.fps,
                                           onCourt: pose.onCourt)
    let probability = walkingProbability(model: walkingModel, signals: frameSignals,
                                         fps: tracks.fps, poseStride: tracks.stride)
    let nearEnd = nearEndSignals(keypoints: pose.keypoints, boxes: pose.boxes,
                                 fps: tracks.fps, speed: frameSignals.speed)
    return EndSignals(fps: tracks.fps, walkingProbability: probability,
                      nearEnd: nearEnd.mapValues { $0.map(toFloat32Precision) })
}
