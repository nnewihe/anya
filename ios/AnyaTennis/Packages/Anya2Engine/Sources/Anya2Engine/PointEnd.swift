// PointEnd.swift  --  port of pipeline/anya2/point_end.py
//
// A point end from POSE ALONE. The "live" score is player activity, vetoed by
// the non-rally union (walking, and near_end's four "between points" signals);
// a point end is the FALLING EDGE of a live run. Not the onset of "looks
// dead": dead time flickers, while a real end is preceded by play.
//
// Evidence and history: docs/anya2/NOTES.md, "point_end.py".
import Foundation

public enum PointEndRules {
    public static let MAX_GAP_S = 0.5            // fill pose occlusions up to this long
    // player activity
    public static let ACTIVITY_MIN_BOX_PX = 8.0  // boxes shorter than this are not measured
    public static let ACTIVITY_KP_CONF = 0.2     // keypoint confidence floor for the limb term
    // the non-rally union: max of these and the walking probability
    public static let UNION_NAMES = ["settle", "turn_away", "stance_drop", "idle_hands"]
    // the live score
    public static let LIVE_SMOOTH_S = 4.0        // smoothing window; longer separates better but blurs the edge
    public static let LIVE_SCALE_PCT = 90.0      // normalise by this percentile of the clip's own score
    public static let LIVE_HI = 0.50             // enter "live" at or above this
    public static let LIVE_LO = 0.35             // ...stay live while at or above this
    public static let LIVE_MIN_S = 2.0           // ignore live runs shorter than this
    // events
    public static let CONFIDENCE_LOOKAHEAD_S = 4.0  // confidence = 1 - mean live score over this window
    public static let REFRACT_S = 6.0               // at most one end per this many seconds
}

/// The non-rally union: per sample, the max of near_end's four signals and the
/// walking probability (point_end.end_signal).
public func nonRallyUnion(_ endSignals: EndSignals) -> [Double] {
    let n = min(endSignals.walkingProbability.count, endSignals.nearEnd["settle"]!.count)
    return (0..<n).map { i in
        PointEndRules.UNION_NAMES.reduce(endSignals.walkingProbability[i]) { best, name in
            let value = endSignals.nearEnd[name]![i]
            return max(best, value.isNaN ? 0 : value)
        }
    }
}

/// [4 slots][N] how fast each player moves, body heights/s; NaN if untracked:
/// max(box travel across the image, fastest wrist/ankle relative to the hip).
/// Deliberately not court speed: far-player court positions carry metres of
/// projection error.
public func playerActivity(_ tracks: PlayerTracks) -> [[Double]] {
    typealias R = PointEndRules
    let maxGap = Int(R.MAX_GAP_S * tracks.fps)
    return (0..<Slots.count).map { slot in
        let keypoints = fillInteriorGaps(tracks.keypoints(slot: slot), maxGap: maxGap)
        let boxes = fillInteriorGaps(tracks.boxes(slot: slot), maxGap: maxGap)
        let bodyHeight = boxHeights(boxes).map { $0 > R.ACTIVITY_MIN_BOX_PX ? $0 : .nan }
        let travel = boxTravelSpeed(boxes: boxes, bodyHeightPx: bodyHeight, fps: tracks.fps)
        let hip = midpointBothRequired(keypointXY(keypoints, Keypoint.leftHip, confFloor: R.ACTIVITY_KP_CONF),
                                       keypointXY(keypoints, Keypoint.rightHip, confFloor: R.ACTIVITY_KP_CONF))
        var fastestLimb = [Double](repeating: .nan, count: tracks.samples)
        for joint in [Keypoint.leftWrist, Keypoint.rightWrist, Keypoint.leftAnkle, Keypoint.rightAnkle] {
            let limb = keypointXY(keypoints, joint, confFloor: R.ACTIVITY_KP_CONF)
            fastestLimb = zip(fastestLimb, limbSpeedRelativeToHip(limb: limb, hip: hip, bodyHeightPx: bodyHeight,
                                                                  fps: tracks.fps)).map(fmaxNaN)
        }
        return zip(travel, fastestLimb).map(fmaxNaN)
    }
}

/// The live score BEFORE normalisation: the busiest player's activity, vetoed
/// by the union, smoothed. This part only looks a bounded way ahead, so it can
/// be computed while recording.
public func smoothedLiveScore(activity: [[Double]], union: [Double], fps: Double,
                              smoothSeconds: Double? = nil) -> [Double] {
    let n = union.count
    let raw: [Double] = (0..<n).map { i in
        let near = Slots.near.reduce(Double.nan) { fmaxNaN($0, activity[$1][i]) }
        let far = Slots.far.reduce(Double.nan) { fmaxNaN($0, activity[$1][i]) }
        return fmaxNaN(near.isNaN ? 0 : near, far.isNaN ? 0 : far) * (1 - union[i])
    }
    let width = max(1, pythonRound((smoothSeconds ?? PointEndRules.LIVE_SMOOTH_S) * fps))
    return convolveSameMean(raw, width: width)
}

/// [N] "a point is in play", ~0..1, where 1 = the clip's 90th percentile.
/// The normalisation needs the WHOLE clip, so live capture applies it at Stop.
public func normaliseLiveScore(_ smoothed: [Double]) -> [Double] {
    let scale = max(percentile(smoothed, PointEndRules.LIVE_SCALE_PCT), 1e-6)
    return smoothed.map { $0 / scale }
}

/// point_end.detect_ends: the falling edge of each live run of at least
/// LIVE_MIN_S; confidence = 1 - the mean live score over the next 4 s.
public func detectPointEnds(live: [Double], fps: Double, enterAt: Double = PointEndRules.LIVE_HI,
                            leaveAt: Double = PointEndRules.LIVE_LO,
                            minimumLiveSeconds: Double = PointEndRules.LIVE_MIN_S) -> [Event] {
    let isLive = hysteresis(live, enterAt: enterAt, leaveAt: leaveAt)
    let minimumRun = max(1, pythonRound(minimumLiveSeconds * fps))
    let lookahead = pythonRound(PointEndRules.CONFIDENCE_LOOKAHEAD_S * fps)
    var ends: [Event] = []
    for run in trueRuns(isLive) where run.stop - run.start >= minimumRun {
        let after = live[run.stop..<min(live.count, run.stop + lookahead)]
        let meanAfter = after.isEmpty ? 1 : mean(Array(after))
        ends.append(Event(t: Double(run.stop) / fps, p: pythonRound(clip(1 - meanAfter, 0, 1), digits: 4),
                          kind: .pointEnd,
                          detail: ["rally_s": pythonRound(Double(run.stop - run.start) / fps, digits: 2)]))
    }
    return keepOnePerWindow(ends, gap: PointEndRules.REFRACT_S, time: { $0.t }, score: { $0.p })
}
