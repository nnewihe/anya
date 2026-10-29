// ServeCommon.swift  --  port of pipeline/anya2/serve_common.py
//
// The steps the near and far serve detectors share exactly. Each detector
// keeps its own features, constants, ready/stillness/toss scoring and timing;
// only the skeleton lives here.
//
// A serve is READY -> TROPHY -> SWING. Both detectors:
//   1. compute a per-sample `trophy` score in [0, 1]
//   2. take every run where trophy >= the detector's minimum (and, if
//      required, the player is in the allowed zone) as ONE candidate, timed
//      at the run's peak                                 [trophyCandidates]
//   3. score the swing: does the RACKET wrist (the one that was lower at the
//      trophy) rise over the head shortly after it        [swing]
//   4. combine:  shape = weighted mean of trophy and ready scores
//                p     = shape * (SWING_FLOOR + (1 - SWING_FLOOR) * swing)
//      so a missing swing can veto, but only softly       [combineServeScore]
//   5. keep the strongest candidate per refractory window, first within each
//      player slot, then across slots (in doubles that decides WHO served)
//
// Image y grows DOWNWARD. Every elevation is written so POSITIVE = HIGHER,
// in body heights ("bh").
import Foundation

/// Seconds -> whole pose samples, rounded as Python does.
@inline(__always) public func samples(_ seconds: Double, fps: Double) -> Int {
    pythonRound(seconds * fps)
}

public struct TrophyCandidate {
    public let runStart: Int   // first sample of the trophy run
    public let peak: Int       // sample where the trophy score peaks
    public let trophy: Double  // trophy score at the peak
}

/// One candidate per run of trophy >= trophyMin (AND `allowed`, if required).
public func trophyCandidates(trophy: [Double], allowed: [Bool], trophyMin: Double,
                             requireCourt: Bool) -> [TrophyCandidate] {
    let score = trophy.map { $0.isNaN ? 0 : $0 }
    let isCandidate = score.indices.map { score[$0] >= trophyMin && (!requireCourt || allowed[$0]) }
    return trueRuns(isCandidate).map { run in
        var peak = run.start
        for i in run.start..<run.stop where score[i] > score[peak] { peak = i }   // argmax: first maximum
        return TrophyCandidate(runStart: run.start, peak: peak, trophy: score[peak])
    }
}

/// What the swing detector needs from a detector's features.
public protocol SwingInputs {
    /// +1 left wrist higher, -1 right, 0 unknown.
    var higherWristSide: [Double] { get }
    /// Each wrist's height above the head, body heights.
    var leftWristAboveHead: [Double] { get }
    var rightWristAboveHead: [Double] { get }
}

public struct Swing {
    public let score: Double          // 0..1: how far the racket wrist cleared the head
    public let contactSeconds: Double?  // time of the racket wrist's highest point
    public let tossIsLeft: Bool       // which arm was tossing (higher wrist at the trophy)
}

/// Did the racket wrist rise over the head shortly after the trophy?
/// The tossing arm carries the higher wrist at the trophy peak; the racket arm
/// is the other. Contact is the racket wrist's highest point in
/// [peak + swingMin, peak + swingMax] -- a position, never a velocity (at
/// 15 Hz the strike spans about two samples).
public func swing<F: SwingInputs>(_ features: F, peak: Int, fps: Double, swingMinSeconds: Double,
                                  swingMaxSeconds: Double, peakAboveHeadBH: Double) -> Swing {
    let tossIsLeft = features.higherWristSide[peak] > 0
    let racketAboveHead = tossIsLeft ? features.rightWristAboveHead : features.leftWristAboveHead
    let n = racketAboveHead.count
    let windowStart = min(n, peak + max(1, samples(swingMinSeconds, fps: fps)))
    let windowStop = min(n, peak + samples(swingMaxSeconds, fps: fps) + 1)
    if windowStop > windowStart {
        var highest: Int? = nil
        for i in windowStart..<windowStop where racketAboveHead[i].isFinite {
            if highest == nil || racketAboveHead[i] > racketAboveHead[highest!] { highest = i }
        }
        if let highest {
            return Swing(score: ramp(racketAboveHead[highest], 0, peakAboveHeadBH),
                         contactSeconds: Double(highest) / fps, tossIsLeft: tossIsLeft)
        }
    }
    return Swing(score: 0, contactSeconds: nil, tossIsLeft: tossIsLeft)
}

/// Weighted shape score, then gated (softly) by the swing.
public func combineServeScore(trophy: Double, ready: Double, swing: Double,
                              trophyWeight: Double, readyWeight: Double, swingFloor: Double) -> Double {
    let shape = (trophyWeight * trophy + readyWeight * ready) / (trophyWeight + readyWeight)
    return shape * (swingFloor + (1 - swingFloor) * swing)
}

/// A scored serve candidate, before it becomes an Event.
public struct ServeCandidate {
    public var t: Double          // point start, seconds
    public var p: Double          // rounded to 4 places, as Python stores it
    public var track: Int
    public var detail: [String: Double]
}
