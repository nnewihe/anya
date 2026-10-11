// FarServe.swift  --  port of pipeline/anya2/far_serve.py
//
// A far-side serve: the same READY -> TROPHY -> SWING idea as the near serve
// (shared skeleton in ServeCommon.swift), adapted to a player 20-40 px tall:
//   - the trophy is ELEVATION ONLY (higher wrist over the head, lower wrist
//     near the shoulders); at this scale the hand split reads backwards
//   - a stillness multiplier: a server stands still before tossing, a player
//     running for a smash does not
//   - a separate toss score, reported with each serve and used by the
//     orchestrator's toss rule, never folded into p
// anya2's own pose helpers, with a 0.20 keypoint confidence floor.
//
// Evidence and history: docs/anya2/NOTES.md, "far_serve.py".
import Foundation

public enum FarServeRules {
    public static let MAX_GAP_S = 0.5            // fill pose occlusions up to this long
    public static let KP_CONF = 0.20             // keypoints at or below this confidence are ignored
    // box sanity (refuse to normalise by a bad box)
    public static let BOX_H_MIN_PX = 12.0, BOX_H_MAX_PX = 120.0
    public static let BOX_ASPECT_MAX = 1.6       // wider than this (w/h) is a merged pair or a fragment
    public static let HEAD_ABOVE_SHO_BH = 0.14   // head height above the shoulders when the head is unseen
    // ready
    public static let READY_HI_MAX_BH = 0.02     // both wrists at or below the shoulder line
    public static let READY_STILL_BH_S = 1.2     // limb speed (relative to hip) counted as quiet
    public static let READY_WIN_S = 0.7          // smoothing window for the ready score
    public static let READY_MEAN_FROM_S = 2.0    // ready = mean over [trophy - 2.0 s, trophy - 0.3 s)
    public static let READY_MEAN_TO_S = 0.3
    // pre-serve stillness (the feet; ready is the arms)
    public static let STILL_FROM_S = 5.0, STILL_TO_S = 1.0   // window before the trophy
    public static let STILL_FULL_BH_S = 0.20     // median travel at or below this = stationary
    public static let STILL_NONE_BH_S = 0.55     // ...at or above this = travelling
    public static let W_STILL = 0.30             // weight of the stillness multiplier
    public static let STILL_UNKNOWN = 0.5        // score when the player was untracked in the window
    // trophy (elevation only), each term dilated by TROPHY_DILATE_S
    public static let TROPHY_HEAD_MIN_BH = -0.02  // higher wrist at/below head height = no credit
    public static let TROPHY_HEAD_FULL_BH = 0.06  // ...this far above = full credit
    public static let TROPHY_LO_MIN_BH = -0.18    // lower wrist this far below the shoulders = none
    public static let TROPHY_LO_FULL_BH = -0.02   // ...level with them = full credit
    public static let TROPHY_DILATE_S = 0.20
    public static let TROPHY_MIN = 0.30
    // swing
    public static let SWING_MIN_S = 0.06, SWING_MAX_S = 0.90
    public static let PEAK_ABOVE_HEAD_BH = 0.04
    // toss score: mean of four ramped parts, read from the tossing arm
    public static let TOSS_PRE_FROM_S = -1.3, TOSS_PRE_TO_S = -0.2   // window (rel. trophy) for "started low"
    public static let TOSS_TOP_FROM_S = -0.3, TOSS_TOP_TO_S = 0.6    // window for the top of the toss
    public static let TOSS_LOW_NONE_BH = -0.22, TOSS_LOW_FULL_BH = -0.42   // lowest wrist vs shoulders
    public static let TOSS_LAT_MIN_BH = 0.07, TOSS_LAT_FULL_BH = 0.17      // wrist sideways from its shoulder
    public static let TOSS_HOLD_MIN_S = 0.13, TOSS_HOLD_FULL_S = 0.42      // time spent above the head
    public static let TOSS_PEAK_MIN_BH = 0.08, TOSS_PEAK_FULL_BH = 0.17    // highest point above the head
    // combination
    public static let W_TROPHY = 0.45, W_READY = 0.20
    public static let SWING_FLOOR = 0.45
    public static let THRESHOLD = 0.75
    public static let REFRACT_S = 3.0
    public static let SERVE_LEAD_S = 0.90        // subtracted from the trophy-RUN START
}

/// Per pose sample, for one far player. Elevations in body heights, + = higher.
public struct FarServeFeatures: SwingInputs {
    public let fps: Double
    public var valid: [Bool]                     // sane box and both wrists seen
    public var ready: [Double]                   // 0..1 arms-low-and-quiet score (smoothed)
    public var trophy: [Double]                  // 0..1 trophy-shape score
    public var leftWristAboveShoulders: [Double], rightWristAboveShoulders: [Double]
    public var leftWristAboveHead: [Double], rightWristAboveHead: [Double]
    public var higherWristAboveShoulders: [Double], lowerWristAboveShoulders: [Double]
    public var higherWristAboveHead: [Double]
    public var higherWristSide: [Double]         // +1 left wrist higher, -1 right, 0 unknown
    public var leftWristSideways: [Double], rightWristSideways: [Double]  // offset from own shoulder
    public var still: [Double]                   // 0..1 limbs quiet
    public var boxTravel: [Double]               // box travel, body heights/s (self_move)
    public var eligible: [Bool]
}

/// Box height in px, NaN where the box is too small, too big or too wide.
private func saneBoxHeight(_ boxes: SampleTable) -> [Double] {
    (0..<boxes.rows).map { i in
        let height = boxes[i, 3] - boxes[i, 1], width = boxes[i, 2] - boxes[i, 0]
        let ok = height > FarServeRules.BOX_H_MIN_PX && height < FarServeRules.BOX_H_MAX_PX
            && width / max(height, 1e-6) < FarServeRules.BOX_ASPECT_MAX
        return ok ? height : .nan
    }
}

/// far_serve.serve_features for one far slot.
public func farServeFeatures(keypoints keypointsIn: SampleTable, boxes boxesIn: SampleTable,
                             fps: Double, eligible: [Bool]?) -> FarServeFeatures {
    typealias R = FarServeRules
    let n = keypointsIn.rows
    let maxGap = Int(R.MAX_GAP_S * fps)
    let keypoints = fillInteriorGaps(keypointsIn, maxGap: maxGap)
    let boxes = fillInteriorGaps(boxesIn, maxGap: maxGap)
    let bodyHeight = saneBoxHeight(boxes)

    func xy(_ k: Int) -> PointTrack { keypointXY(keypoints, k, confFloor: R.KP_CONF) }
    let leftShoulder = xy(Keypoint.leftShoulder), rightShoulder = xy(Keypoint.rightShoulder)
    let leftWrist = xy(Keypoint.leftWrist), rightWrist = xy(Keypoint.rightWrist)
    let shoulders = midpointBothRequired(leftShoulder, rightShoulder)
    let hips = midpointBothRequired(xy(Keypoint.leftHip), xy(Keypoint.rightHip))
    let head = midpointBothRequired(midpointBothRequired(xy(Keypoint.leftEar), xy(Keypoint.rightEar)),
                                    xy(Keypoint.nose))

    var f = FarServeFeatures(
        fps: fps, valid: [], ready: [], trophy: [],
        leftWristAboveShoulders: [], rightWristAboveShoulders: [],
        leftWristAboveHead: [], rightWristAboveHead: [],
        higherWristAboveShoulders: [Double](repeating: .nan, count: n),
        lowerWristAboveShoulders: [Double](repeating: .nan, count: n),
        higherWristAboveHead: [Double](repeating: .nan, count: n),
        higherWristSide: [Double](repeating: 0, count: n),
        leftWristSideways: (0..<n).map { abs(leftWrist.xs[$0] - leftShoulder.xs[$0]) / bodyHeight[$0] },
        rightWristSideways: (0..<n).map { abs(rightWrist.xs[$0] - rightShoulder.xs[$0]) / bodyHeight[$0] },
        still: [], boxTravel: boxTravelSpeed(boxes: boxes, bodyHeightPx: bodyHeight, fps: fps),
        eligible: eligible ?? [Bool](repeating: true, count: n))
    var bothWrists = [Bool](repeating: false, count: n)
    for i in 0..<n {
        let headY = head.ys[i].isFinite ? head.ys[i] : shoulders.ys[i] - R.HEAD_ABOVE_SHO_BH * bodyHeight[i]
        let leftElevation = (shoulders.ys[i] - leftWrist.ys[i]) / bodyHeight[i]
        let rightElevation = (shoulders.ys[i] - rightWrist.ys[i]) / bodyHeight[i]
        f.leftWristAboveShoulders.append(leftElevation)
        f.rightWristAboveShoulders.append(rightElevation)
        f.leftWristAboveHead.append((headY - leftWrist.ys[i]) / bodyHeight[i])
        f.rightWristAboveHead.append((headY - rightWrist.ys[i]) / bodyHeight[i])
        bothWrists[i] = leftElevation.isFinite && rightElevation.isFinite
        if bothWrists[i] {
            f.higherWristAboveShoulders[i] = fmaxNaN(leftElevation, rightElevation)
            f.lowerWristAboveShoulders[i] = fminNaN(leftElevation, rightElevation)
            f.higherWristAboveHead[i] = fmaxNaN(f.leftWristAboveHead[i], f.rightWristAboveHead[i])
            f.higherWristSide[i] = leftElevation >= rightElevation ? 1 : -1
        }
    }
    f.valid = (0..<n).map { bodyHeight[$0].isFinite && bothWrists[$0] }

    // ready: wrists carried low * limbs quiet, smoothed. Missing samples are
    // skipped, not counted as 0: a tracking dropout is not evidence that the
    // player was out of the ready stance (QUIRKS #9).
    let fastestWrist = zip(limbSpeedRelativeToHip(limb: leftWrist, hip: hips, bodyHeightPx: bodyHeight, fps: fps),
                           limbSpeedRelativeToHip(limb: rightWrist, hip: hips, bodyHeightPx: bodyHeight, fps: fps)).map(fmaxNaN)
    f.still = fastestWrist.map { 1 - ramp($0, R.READY_STILL_BH_S * 0.5, R.READY_STILL_BH_S * 1.6) }
    let carriedLow = f.higherWristAboveShoulders.map { 1 - ramp($0, R.READY_HI_MAX_BH, R.READY_HI_MAX_BH + 0.14) }
    f.ready = nanSkippingMovingMean((0..<n).map { carriedLow[$0] * f.still[$0] },
                                    width: max(1, samples(R.READY_WIN_S, fps: fps)))

    // trophy: (higher wrist over head) * (lower wrist near shoulders)
    let dilate = max(1, samples(R.TROPHY_DILATE_S, fps: fps) * 2 + 1)
    let overHead = runningMax(ramp(f.higherWristAboveHead, R.TROPHY_HEAD_MIN_BH, R.TROPHY_HEAD_FULL_BH), width: dilate)
    let otherNearShoulders = runningMax(ramp(f.lowerWristAboveShoulders, R.TROPHY_LO_MIN_BH, R.TROPHY_LO_FULL_BH), width: dilate)
    f.trophy = (0..<n).map { overHead[$0] * otherNearShoulders[$0] }
    return f
}

/// Mean ready score over [peak - READY_MEAN_FROM_S, peak - READY_MEAN_TO_S).
private func readyBefore(_ f: FarServeFeatures, peak: Int) -> Double {
    let start = max(0, peak - samples(FarServeRules.READY_MEAN_FROM_S, fps: f.fps))
    let stop = max(1, peak - samples(FarServeRules.READY_MEAN_TO_S, fps: f.fps))
    let window = start < stop ? finiteValues(f.ready[start..<stop]) : []
    return window.isEmpty ? 0 : mean(window)
}

/// 1 = stood still before the trophy, 0 = was travelling; STILL_UNKNOWN if untracked.
private func stillBefore(_ f: FarServeFeatures, peak: Int) -> Double {
    let start = max(0, peak - samples(FarServeRules.STILL_FROM_S, fps: f.fps))
    let stop = max(1, peak - samples(FarServeRules.STILL_TO_S, fps: f.fps))
    let travel = start < stop ? finiteValues(f.boxTravel[start..<stop]) : []
    guard !travel.isEmpty else { return FarServeRules.STILL_UNKNOWN }
    return 1 - ramp(median(travel), FarServeRules.STILL_FULL_BH_S, FarServeRules.STILL_NONE_BH_S)
}

/// Did the tossing arm do what a toss does? Separate from p. Mean of four
/// 0..1 parts relative to the trophy peak:
///   low   the wrist started well below the shoulders
///   lat   at the top, the wrist is offset sideways from its shoulder
///   hold  the wrist stayed above the head for a while
///   peak  how far above the head it got
/// nil when the arm was not tracked -- which must not be read as a bad toss.
public func farServeTossScore(_ f: FarServeFeatures, peak: Int, tossIsLeft: Bool)
    -> (toss: Double, parts: [String: Double])? {
    typealias R = FarServeRules
    let elevation = tossIsLeft ? f.leftWristAboveShoulders : f.rightWristAboveShoulders
    let aboveHead = tossIsLeft ? f.leftWristAboveHead : f.rightWristAboveHead
    let sideways = tossIsLeft ? f.leftWristSideways : f.rightWristSideways
    func window(_ x: [Double], _ fromSeconds: Double, _ toSeconds: Double) -> [Double] {
        let start = max(0, peak + pythonRound(fromSeconds * f.fps))
        let stop = min(x.count, peak + pythonRound(toSeconds * f.fps))
        return start < stop ? finiteValues(x[start..<stop]) : []
    }
    let beforeRise = window(elevation, R.TOSS_PRE_FROM_S, R.TOSS_PRE_TO_S)
    let topSideways = window(sideways, R.TOSS_TOP_FROM_S, R.TOSS_TOP_TO_S)
    let topAboveHead = window(aboveHead, R.TOSS_TOP_FROM_S, R.TOSS_TOP_TO_S)
    guard !beforeRise.isEmpty, !topSideways.isEmpty, !topAboveHead.isEmpty else { return nil }
    let low = 1 - ramp(beforeRise.min()!, R.TOSS_LOW_FULL_BH, R.TOSS_LOW_NONE_BH)
    let lat = ramp(median(topSideways), R.TOSS_LAT_MIN_BH, R.TOSS_LAT_FULL_BH)
    let hold = ramp(Double(topAboveHead.filter { $0 > 0 }.count) / f.fps, R.TOSS_HOLD_MIN_S, R.TOSS_HOLD_FULL_S)
    let peakHeight = ramp(topAboveHead.max()!, R.TOSS_PEAK_MIN_BH, R.TOSS_PEAK_FULL_BH)
    let parts = ["low": low, "lat": lat, "hold": hold, "peak": peakHeight]
    return (pythonRound(mean([low, lat, hold, peakHeight]), digits: 4),
            parts.mapValues { pythonRound($0, digits: 3) })
}

/// far_serve.detect_serves: score every trophy candidate; keep p >= threshold.
public func detectFarServes(_ f: FarServeFeatures, threshold: Double = FarServeRules.THRESHOLD,
                            requireCourt: Bool = true, track: Int, leadSeconds: Double? = nil,
                            refractSeconds: Double? = nil, stillWeight: Double? = nil) -> [ServeCandidate] {
    typealias R = FarServeRules
    let fps = f.fps
    let wStill = stillWeight ?? R.W_STILL
    var serves: [ServeCandidate] = []
    for candidate in trophyCandidates(trophy: f.trophy, allowed: f.eligible,
                                      trophyMin: R.TROPHY_MIN, requireCourt: requireCourt) {
        let ready = readyBefore(f, peak: candidate.peak)
        let still = stillBefore(f, peak: candidate.peak)
        let swingResult = swing(f, peak: candidate.peak, fps: fps, swingMinSeconds: R.SWING_MIN_S,
                                swingMaxSeconds: R.SWING_MAX_S, peakAboveHeadBH: R.PEAK_ABOVE_HEAD_BH)
        var p = combineServeScore(trophy: candidate.trophy, ready: ready, swing: swingResult.score,
                                  trophyWeight: R.W_TROPHY, readyWeight: R.W_READY, swingFloor: R.SWING_FLOOR)
        p *= (1 - wStill) + wStill * still
        guard p >= threshold else { continue }
        var detail: [String: Double] = [
            "trophy": pythonRound(candidate.trophy, digits: 4), "swing": pythonRound(swingResult.score, digits: 4),
            "ready": pythonRound(ready, digits: 4), "still": pythonRound(still, digits: 4),
            "t_trophy": pythonRound(Double(candidate.peak) / fps, digits: 3),
        ]
        if let contact = swingResult.contactSeconds { detail["t_contact"] = pythonRound(contact, digits: 3) }
        if let toss = farServeTossScore(f, peak: candidate.peak, tossIsLeft: swingResult.tossIsLeft) {
            detail["toss"] = toss.toss
            for (name, value) in toss.parts { detail["toss_\(name)"] = value }
        }
        serves.append(ServeCandidate(t: Double(candidate.runStart) / fps - (leadSeconds ?? R.SERVE_LEAD_S),
                                     p: pythonRound(p, digits: 4), track: track, detail: detail))
    }
    return keepOnePerWindow(serves, gap: refractSeconds ?? R.REFRACT_S, time: { $0.t }, score: { $0.p })
}

/// far_serve.detect_video: each far slot on its own, then one serve per
/// refractory window across both slots.
public func detectFarServes(tracks: PlayerTracks, threshold: Double = FarServeRules.THRESHOLD,
                            requireCourt: Bool = true, leadSeconds: Double? = nil,
                            refractSeconds: Double? = nil, stillWeight: Double? = nil) -> [Event] {
    var candidates: [ServeCandidate] = []
    for slot in Slots.far where tracks.wasTracked(slot: slot) {
        let features = farServeFeatures(keypoints: tracks.keypoints(slot: slot), boxes: tracks.boxes(slot: slot),
                                        fps: tracks.fps, eligible: tracks.eligible(slot: slot))
        candidates += detectFarServes(features, threshold: threshold, requireCourt: requireCourt, track: slot,
                                      leadSeconds: leadSeconds, refractSeconds: refractSeconds, stillWeight: stillWeight)
    }
    let kept = keepOnePerWindow(candidates, gap: refractSeconds ?? FarServeRules.REFRACT_S,
                                time: { $0.t }, score: { $0.p })
    return kept.map { Event(t: $0.t, p: $0.p, kind: .farServe, track: $0.track, detail: $0.detail) }
}
