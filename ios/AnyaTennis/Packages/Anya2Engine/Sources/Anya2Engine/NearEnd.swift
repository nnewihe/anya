// NearEnd.swift  --  port of pipeline/near_end.py (near_signals)
//
// Four "the point is over" signals read off the near player's pose:
//   settle       the player has gone quiet (feet and limbs)
//   turn_away    the player has turned to face the camera (shoulder order)
//   stance_drop  hands dropped AND legs straightened out of the ready crouch
//   idle_hands   a hand in the pocket at the hip
// Each is in [0, 1], 0 wherever the player is untracked. They use the walking
// classifier's pose helpers on purpose: they read the same near-player track.
//
// Evidence and history: docs/anya2/NOTES.md, "near_end.py".
import Foundation

public enum NearEndRules {
    public static let SIGNAL_NAMES = ["settle", "turn_away", "stance_drop", "idle_hands"]
    public static let MAX_GAP_S = 0.5            // fill occlusions up to this long
    public static let SETTLE_SPEED_MPS = 0.55    // court speed scoring 0 (a walk is 1.0-1.5 m/s)
    public static let SETTLE_LIMB_BH_S = 0.75    // wrist/ankle speed relative to the hip, body heights/s
    public static let SETTLE_WIN_S = 1.2
    public static let TURN_MIN_SHO_BH = 0.055    // shoulders closer than this (edge-on): no read
    public static let TURN_FULL_BH = 0.14        // shoulder separation for a fully confident read
    public static let TURN_WIN_S = 0.8
    public static let READY_WRIST_BH = 0.06      // highest wrist this far ABOVE the hip line = ready
    public static let DROP_WRIST_BH = -0.06      // ...this far below = hands dropped
    public static let READY_KNEE_DEG = 152.0     // mean knee angle of an athletic crouch
    public static let STRAIGHT_KNEE_DEG = 172.0  // ...and of standing upright
    public static let STANCE_WIN_S = 0.8
    public static let POCKET_R_BH = 0.13         // wrist this close to the same-side hip, and below it
    public static let IDLE_WIN_S = 0.6
}

/// The keypoints near_end reads, as image positions.
private struct NearEndJoints {
    let leftShoulder, rightShoulder, leftWrist, rightWrist: PointTrack
    let leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle: PointTrack
    let hip: PointTrack

    init(_ keypoints: SampleTable) {
        func xy(_ k: Int) -> PointTrack { keypointXYWalking(keypoints, k) }
        leftShoulder = xy(Keypoint.leftShoulder); rightShoulder = xy(Keypoint.rightShoulder)
        leftWrist = xy(Keypoint.leftWrist); rightWrist = xy(Keypoint.rightWrist)
        leftHip = xy(Keypoint.leftHip); rightHip = xy(Keypoint.rightHip)
        leftKnee = xy(Keypoint.leftKnee); rightKnee = xy(Keypoint.rightKnee)
        leftAnkle = xy(Keypoint.leftAnkle); rightAnkle = xy(Keypoint.rightAnkle)
        hip = midpointWithFallback(leftHip, rightHip)
    }
}

/// near_end.near_signals: the four signals for one near-player pose track.
/// `speed` is the court speed (m/s) from the walking frame signals, the same
/// number the walking classifier uses; without it, `settle` uses limbs alone.
public func nearEndSignals(keypoints keypointsIn: SampleTable, boxes boxesIn: SampleTable,
                           fps: Double, speed: [Double]?) -> [String: [Double]] {
    let n = keypointsIn.rows
    let maxGap = Int(NearEndRules.MAX_GAP_S * fps)
    let keypoints = fillGapsBetweenSeen(keypointsIn, maxGap: maxGap)
    let boxes = fillGapsBetweenSeen(boxesIn, maxGap: maxGap)
    let bodyHeight = boxHeights(boxes).map { $0 > 1 ? $0 : .nan }
    let valid = bodyHeight.map(\.isFinite)
    let j = NearEndJoints(keypoints)

    func smooth(_ x: [Double], _ windowSeconds: Double) -> [Double] {
        nanSkippingMovingMean(x, width: max(1, pythonRound(windowSeconds * fps)))
    }
    func limbSpeed(_ point: PointTrack) -> [Double] {
        limbSpeedRelativeToHip(limb: point, hip: j.hip, bodyHeightPx: bodyHeight, fps: fps)
    }

    // settle: mean of "feet quiet" and "limbs quiet" (the two drop out
    // independently at 15 Hz, and the mean keeps whichever is readable).
    let fastestLimb = zip(zip(limbSpeed(j.leftWrist), limbSpeed(j.rightWrist)).map(fmaxNaN),
                          zip(limbSpeed(j.leftAnkle), limbSpeed(j.rightAnkle)).map(fmaxNaN)).map(fmaxNaN)
    let limbsQuiet = ramp(fastestLimb, NearEndRules.SETTLE_LIMB_BH_S, 0)
    let settleRaw = speed.map { nanMeanOfTwo(ramp($0, NearEndRules.SETTLE_SPEED_MPS, 0), limbsQuiet) } ?? limbsQuiet

    // turn_away: back to the camera, the right shoulder is image-right of the
    // left; turned round, the order flips. Edge-on shoulders give no read.
    let turnRaw: [Double] = (0..<n).map { i in
        let shoulderDx = (j.rightShoulder.xs[i] - j.leftShoulder.xs[i]) / bodyHeight[i]
        let readConfidence = ramp(abs(shoulderDx), NearEndRules.TURN_MIN_SHO_BH, NearEndRules.TURN_FULL_BH)
        let facingCamera = ramp(-shoulderDx, 0, NearEndRules.TURN_FULL_BH)
        return clip(readConfidence * facingCamera, 0, 1)
    }

    // stance_drop: hands dropped AND legs straight (straight legs alone mean nothing).
    // Image y grows downward, so hip_y - wrist_y > 0 means above the hip.
    let handsLow: [Double] = (0..<n).map { i in
        let highestWristAboveHip = (j.hip.ys[i] - fminNaN(j.leftWrist.ys[i], j.rightWrist.ys[i])) / bodyHeight[i]
        return ramp(highestWristAboveHip, NearEndRules.READY_WRIST_BH, NearEndRules.DROP_WRIST_BH)
    }
    let kneeAngle = nanMeanOfTwo(jointAngleDegrees(j.leftHip, j.leftKnee, j.leftAnkle),
                                 jointAngleDegrees(j.rightHip, j.rightKnee, j.rightAnkle))
    let legsStraight = ramp(kneeAngle, NearEndRules.READY_KNEE_DEG, NearEndRules.STRAIGHT_KNEE_DEG)
    let stanceRaw = zip(handsLow, legsStraight).map(fminNaN)

    // idle_hands: hand to pocket -- at the hip AND dropped below it.
    let leftHipDistance = distance(j.leftWrist, j.leftHip, over: bodyHeight)
    let rightHipDistance = distance(j.rightWrist, j.rightHip, over: bodyHeight)
    let pocket: [Double] = (0..<n).map { i in
        let leftBelowHip = (j.leftWrist.ys[i] - j.leftHip.ys[i]) / bodyHeight[i]
        let rightBelowHip = (j.rightWrist.ys[i] - j.rightHip.ys[i]) / bodyHeight[i]
        return fmaxNaN(ramp(leftHipDistance[i], NearEndRules.POCKET_R_BH, 0) * ramp(leftBelowHip, 0, 0.05),
                       ramp(rightHipDistance[i], NearEndRules.POCKET_R_BH, 0) * ramp(rightBelowHip, 0, 0.05))
    }

    let raw: [String: [Double]] = [
        "settle": smooth(settleRaw, NearEndRules.SETTLE_WIN_S),
        "turn_away": smooth(turnRaw, NearEndRules.TURN_WIN_S),
        "stance_drop": smooth(stanceRaw, NearEndRules.STANCE_WIN_S),
        "idle_hands": smooth(pocket, NearEndRules.IDLE_WIN_S),
    ]
    // Missing evidence is 0, and nothing is claimed where there is no player.
    return raw.mapValues { values in
        (0..<n).map { i in valid[i] && values[i].isFinite ? clip(values[i], 0, 1) : 0 }
    }
}
