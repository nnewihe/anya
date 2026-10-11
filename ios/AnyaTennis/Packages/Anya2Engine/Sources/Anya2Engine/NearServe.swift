// NearServe.swift  --  port of pipeline/anya2/near_serve.py
//
// A near-side serve from POSE ALONE: a ready stance (hands together on the
// grip, wrists low, still), then a trophy (one wrist high over the head, the
// other near the shoulders, hands apart), then a swing (the racket wrist
// rising over the head). Shared skeleton: ServeCommon.swift.
//
// The near player is big in the frame, so this uses the walking classifier's
// pose helpers (same near-player conventions as near_end).
//
// Evidence and history: docs/anya2/NOTES.md, "near_serve.py".
import Foundation

public enum NearServeRules {
    public static let MAX_GAP_S = 0.5            // fill pose occlusions up to this long
    public static let HEAD_ABOVE_SHO_BH = 0.14   // head height above the shoulders when the head is unseen
    // ready
    public static let TOGETHER_BH = 0.10         // wrists this close = hands together on the grip
    public static let SPLIT_BH = 0.26            // ...this far apart = definitely split
    public static let READY_HI_MAX_BH = 0.10     // higher wrist no more than this above the shoulders
    public static let READY_STILL_BH_S = 0.9     // limb speed (relative to hip) counted as quiet
    public static let READY_WIN_S = 0.7          // smoothing window for the ready score
    // trophy
    public static let TROPHY_ABOVE_HEAD_BH = 0.058  // higher wrist this far over the head = full credit
    public static let TROPHY_HEAD_MIN_BH = -0.02    // ...at/below head height = no credit
    public static let TROPHY_LO_MIN_BH = -0.13      // lower wrist this far below the shoulders = no credit
    public static let TROPHY_LO_FULL_BH = -0.052    // ...near the shoulders = full credit
    public static let TROPHY_SPLIT_MIN_BH = 0.149   // hands must be at least this far apart to score
    public static let TROPHY_SPLIT_RANGE_BH = 0.12  // ...and get full credit this much further apart
    public static let TROPHY_DILATE_S = 0.20        // +/- window each term is max-dilated over
    public static let TROPHY_MIN = 0.35             // trophy score that makes a sample a candidate
    // swing
    public static let SWING_MIN_S = 0.06         // contact search starts this long after the trophy
    public static let SWING_MAX_S = 0.90         // ...and ends this long after
    public static let PEAK_ABOVE_HEAD_BH = 0.06  // racket wrist this far over the head = full swing
    // ready look-back
    public static let READY_BACK_MIN_S = 0.20
    public static let READY_BACK_MAX_S = 6.00
    // the serve zone, on the server's feet
    public static let SERVE_ZONE_MIN_FT = -3.5   // up to 3.5 ft behind the baseline
    public static let SERVE_ZONE_MAX_FT = 1.0    // ...and up to 1 ft inside the court
    // scoring
    public static let W_TROPHY = 0.45, W_READY = 0.20
    public static let SWING_FLOOR = 0.45         // score multiplier kept with NO swing evidence
    public static let THRESHOLD = 0.70           // report serves with p >= this
    public static let REFRACT_S = 3.0            // at most one serve per this many seconds
    public static let SERVE_LEAD_S = 1.75        // point start = trophy peak minus this (label convention)
}

/// Per pose sample, for one near player. Elevations in body heights, + = higher.
public struct NearServeFeatures: SwingInputs {
    public let fps: Double
    public var valid: [Bool]                 // box usable and both wrists seen
    public var ready: [Double]               // 0..1 ready-stance score (smoothed)
    public var trophy: [Double]              // 0..1 trophy-shape score
    public var leftWristAboveShoulders: [Double], rightWristAboveShoulders: [Double]
    public var leftWristAboveHead: [Double], rightWristAboveHead: [Double]
    public var higherWristAboveShoulders: [Double], lowerWristAboveShoulders: [Double]
    public var higherWristAboveHead: [Double]
    public var higherWristSide: [Double]     // +1 left wrist higher, -1 right, 0 unknown
    public var wristGap: [Double]            // distance between the wrists, body heights
    public var still: [Double]               // 0..1 limbs quiet
    public var inServeZone: [Bool]           // in the serve zone AND eligible
}

/// near_serve.serve_features for one near slot. `courtY` (feet, metres)
/// enables the serve-zone gate; `eligible` is ANDed into it.
public func nearServeFeatures(keypoints keypointsIn: SampleTable, boxes boxesIn: SampleTable, fps: Double,
                              courtY: [Double]?, eligible: [Bool]?) -> NearServeFeatures {
    typealias R = NearServeRules
    let n = keypointsIn.rows
    let maxGap = Int(R.MAX_GAP_S * fps)
    let keypoints = fillGapsBetweenSeen(keypointsIn, maxGap: maxGap)
    let boxes = fillGapsBetweenSeen(boxesIn, maxGap: maxGap)
    let bodyHeight = boxHeights(boxes).map { $0 > 1 ? $0 : .nan }

    func xy(_ k: Int) -> PointTrack { keypointXYWalking(keypoints, k) }
    let leftWrist = xy(Keypoint.leftWrist), rightWrist = xy(Keypoint.rightWrist)
    let shoulders = midpointWithFallback(xy(Keypoint.leftShoulder), xy(Keypoint.rightShoulder))
    let hips = midpointWithFallback(xy(Keypoint.leftHip), xy(Keypoint.rightHip))
    let head = midpointWithFallback(midpointWithFallback(xy(Keypoint.leftEar), xy(Keypoint.rightEar)),
                                    xy(Keypoint.nose))

    var f = NearServeFeatures(
        fps: fps, valid: [], ready: [], trophy: [],
        leftWristAboveShoulders: [], rightWristAboveShoulders: [],
        leftWristAboveHead: [], rightWristAboveHead: [],
        higherWristAboveShoulders: [Double](repeating: .nan, count: n),
        lowerWristAboveShoulders: [Double](repeating: .nan, count: n),
        higherWristAboveHead: [Double](repeating: .nan, count: n),
        higherWristSide: [Double](repeating: 0, count: n),
        wristGap: distance(leftWrist, rightWrist, over: bodyHeight), still: [], inServeZone: [])
    var bothWrists = [Bool](repeating: false, count: n)
    for i in 0..<n {
        // A head we cannot see sits a fixed height above the shoulders.
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

    let fastestWrist = zip(limbSpeedRelativeToHip(limb: leftWrist, hip: hips, bodyHeightPx: bodyHeight, fps: fps),
                           limbSpeedRelativeToHip(limb: rightWrist, hip: hips, bodyHeightPx: bodyHeight, fps: fps)).map(fmaxNaN)
    f.still = fastestWrist.map { 1 - ramp($0, R.READY_STILL_BH_S * 0.5, R.READY_STILL_BH_S * 1.6) }
    let together = f.wristGap.map { 1 - ramp($0, R.TOGETHER_BH, R.SPLIT_BH) }
    let carriedLow = f.higherWristAboveShoulders.map { 1 - ramp($0, R.READY_HI_MAX_BH, R.READY_HI_MAX_BH + 0.14) }
    let readyRaw = (0..<n).map { together[$0] * carriedLow[$0] * f.still[$0] }
    f.ready = nanSkippingMovingMean(readyRaw, width: max(1, samples(R.READY_WIN_S, fps: fps)))

    let dilate = max(1, samples(R.TROPHY_DILATE_S, fps: fps) * 2 + 1)
    let overHead = runningMax(ramp(f.higherWristAboveHead, R.TROPHY_HEAD_MIN_BH, R.TROPHY_ABOVE_HEAD_BH), width: dilate)
    let otherNearShoulders = runningMax(ramp(f.lowerWristAboveShoulders, R.TROPHY_LO_MIN_BH, R.TROPHY_LO_FULL_BH), width: dilate)
    let handsApart = runningMax(ramp(f.wristGap, R.TROPHY_SPLIT_MIN_BH, R.TROPHY_SPLIT_MIN_BH + R.TROPHY_SPLIT_RANGE_BH), width: dilate)
    f.trophy = (0..<n).map { overHead[$0] * otherNearShoulders[$0] * handsApart[$0] }

    f.inServeZone = (0..<n).map { i in
        var inZone = true
        if let courtY {
            // NaN feet answer false: a server we cannot place is excluded.
            inZone = courtY[i] >= R.SERVE_ZONE_MIN_FT * Court.FT_TO_M && courtY[i] <= R.SERVE_ZONE_MAX_FT * Court.FT_TO_M
        }
        return inZone && (eligible?[i] ?? true)
    }
    return f
}

/// near_serve.detect_serves: score every trophy candidate; keep p >= threshold.
public func detectNearServes(_ f: NearServeFeatures, threshold: Double = NearServeRules.THRESHOLD,
                             requireCourt: Bool = true, track: Int,
                             leadSeconds: Double? = nil, refractSeconds: Double? = nil) -> [ServeCandidate] {
    typealias R = NearServeRules
    let fps = f.fps
    let readyFrom = samples(R.READY_BACK_MAX_S, fps: fps), readyTo = samples(R.READY_BACK_MIN_S, fps: fps)
    var serves: [ServeCandidate] = []
    for candidate in trophyCandidates(trophy: f.trophy, allowed: f.inServeZone,
                                      trophyMin: R.TROPHY_MIN, requireCourt: requireCourt) {
        let first = max(0, candidate.peak - readyFrom), last = max(0, candidate.peak - readyTo)
        let lookback = first < last ? finiteValues(f.ready[first..<last]) : []
        let ready = lookback.max() ?? 0
        let swingResult = swing(f, peak: candidate.peak, fps: fps, swingMinSeconds: R.SWING_MIN_S,
                                swingMaxSeconds: R.SWING_MAX_S, peakAboveHeadBH: R.PEAK_ABOVE_HEAD_BH)
        let p = combineServeScore(trophy: candidate.trophy, ready: ready, swing: swingResult.score,
                                  trophyWeight: R.W_TROPHY, readyWeight: R.W_READY, swingFloor: R.SWING_FLOOR)
        guard p >= threshold else { continue }
        // Point start: the trophy peak minus the label lead (NOTES: "Near-serve trophy anchor").
        let start = Double(candidate.peak) / fps - (leadSeconds ?? R.SERVE_LEAD_S)
        var detail: [String: Double] = [
            "trophy": pythonRound(candidate.trophy, digits: 4), "swing": pythonRound(swingResult.score, digits: 4),
            "ready": pythonRound(ready, digits: 4), "t_trophy": pythonRound(Double(candidate.peak) / fps, digits: 3),
            "toss_arm_left": swingResult.tossIsLeft ? 1 : 0,
        ]
        if let contact = swingResult.contactSeconds { detail["t_contact"] = pythonRound(contact, digits: 3) }
        serves.append(ServeCandidate(t: start, p: pythonRound(p, digits: 4), track: track, detail: detail))
    }
    return keepOnePerWindow(serves, gap: refractSeconds ?? R.REFRACT_S, time: { $0.t }, score: { $0.p })
}

/// near_serve.detect_video: each near slot scored on its own (never averaged:
/// a serve is one player's action), then the strongest serve kept per
/// refractory window across both slots -- in doubles, that decides who served.
public func detectNearServes(tracks: PlayerTracks, threshold: Double = NearServeRules.THRESHOLD,
                             requireCourt: Bool = true, leadSeconds: Double? = nil,
                             refractSeconds: Double? = nil) -> [Event] {
    var candidates: [ServeCandidate] = []
    for slot in Slots.near where tracks.wasTracked(slot: slot) {
        let features = nearServeFeatures(keypoints: tracks.keypoints(slot: slot), boxes: tracks.boxes(slot: slot),
                                         fps: tracks.fps, courtY: tracks.court(slot: slot, axis: 1),
                                         eligible: tracks.eligible(slot: slot))
        candidates += detectNearServes(features, threshold: threshold, requireCourt: requireCourt, track: slot,
                                       leadSeconds: leadSeconds, refractSeconds: refractSeconds)
    }
    let kept = keepOnePerWindow(candidates, gap: refractSeconds ?? NearServeRules.REFRACT_S,
                                time: { $0.t }, score: { $0.p })
    return kept.map { Event(t: $0.t, p: $0.p, kind: .nearServe, track: $0.track, detail: $0.detail) }
}
