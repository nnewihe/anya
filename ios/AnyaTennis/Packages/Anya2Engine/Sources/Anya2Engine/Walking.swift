// Walking.swift  --  port of walking/features.py + walking/predict.py
//
// The near-player walking classifier: per-frame signals, multi-scale window
// features, and the gradient-boosted trees (exported to
// Resources/walking_model.json by tools/export_walking_model.py).
//
// Only the walking PROBABILITY is ported: point_end reads `prob`; the
// post-processed walking mask is never used downstream.
//
// Gaps up to 0.5 s are filled (single-frame occlusions); longer runs stay NaN
// and flow through to the features as NaN, which the trees handle natively.
import Foundation

public enum WalkingFeatureRules {
    public static let MAX_GAP_S = 0.5                         // fill detection gaps no longer than this
    public static let WINDOWS_S: [Double] = [0.5, 1, 2, 4, 8] // the window scales, seconds
    public static let CADENCE_BAND = (low: 0.7, high: 4.0)    // Hz: walking through sprinting step rates
    public static let SPEED_SMOOTH_S = 0.25                   // smoothing before differentiating position
}

/// The per-frame quantities the window features and near_end read
/// (the subset of walking.features.frame_signals that is used).
public struct WalkingFrameSignals {
    public var courtX: [Double], courtY: [Double]       // feet position, metres
    public var smoothedX: [Double], smoothedY: [Double] // cx_s, cy_s
    public var velocityX: [Double], velocityY: [Double]
    public var speed: [Double]                           // m/s, also near_end's "feet quiet" input
    public var acceleration: [Double]
    public var ankleSeparation: [Double]                 // body heights
    public var ankleVerticalGap: [Double]                // ank_dy
    public var hipAboveAnkle: [Double]                   // crouch depth
    public var kneeAngle: [Double]
    public var wristSpeedRelativeToHip: [Double]
    public var ankleSpeedRelativeToHip: [Double]
    public var valid: [Double]                           // 1 where the feet were placed on court
    public var onCourt: [Double]
}

/// walking.features.frame_signals. `homography` is the FIXED calibration
/// homography -- the walking classifier never used the camera track.
public func walkingFrameSignals(keypoints keypointsIn: SampleTable, boxes boxesIn: SampleTable,
                                homography: Homography, fps: Double, onCourt: [Double]?) -> WalkingFrameSignals {
    let n = keypointsIn.rows
    let maxGap = Int(WalkingFeatureRules.MAX_GAP_S * fps)
    let keypoints = fillGapsBetweenSeen(keypointsIn, maxGap: maxGap)
    let boxes = fillGapsBetweenSeen(boxesIn, maxGap: maxGap)
    let bodyHeight = boxHeights(boxes).map { $0 > 1 ? $0 : .nan }

    let leftAnkle = keypointXYWalking(keypoints, Keypoint.leftAnkle)
    let rightAnkle = keypointXYWalking(keypoints, Keypoint.rightAnkle)
    let leftHip = keypointXYWalking(keypoints, Keypoint.leftHip), rightHip = keypointXYWalking(keypoints, Keypoint.rightHip)
    let leftKnee = keypointXYWalking(keypoints, Keypoint.leftKnee), rightKnee = keypointXYWalking(keypoints, Keypoint.rightKnee)
    let leftWrist = keypointXYWalking(keypoints, Keypoint.leftWrist), rightWrist = keypointXYWalking(keypoints, Keypoint.rightWrist)

    // Feet: the ankle midpoint, falling back to the box's bottom-centre.
    var feet = midpointWithFallback(leftAnkle, rightAnkle)
    for i in 0..<n where feet.xs[i].isNaN {
        feet.xs[i] = 0.5 * (boxes[i, 0] + boxes[i, 2])
        feet.ys[i] = boxes[i, 3]
    }
    var courtX = [Double](repeating: .nan, count: n), courtY = courtX
    for i in 0..<n where feet.xs[i].isFinite && feet.ys[i].isFinite {
        (courtX[i], courtY[i]) = homography.apply(x: feet.xs[i], y: feet.ys[i])
    }
    let hip = midpointWithFallback(leftHip, rightHip)

    let width = max(1, pythonRound(WalkingFeatureRules.SPEED_SMOOTH_S * fps))
    let smoothedX = nanSkippingMovingMean(courtX, width: width)
    let smoothedY = nanSkippingMovingMean(courtY, width: width)
    var velocityX = [Double](repeating: .nan, count: n), velocityY = velocityX
    for i in 1..<max(n, 1) {
        velocityX[i] = (smoothedX[i] - smoothedX[i - 1]) * fps
        velocityY[i] = (smoothedY[i] - smoothedY[i - 1]) * fps
    }
    let speed = zip(velocityX, velocityY).map { ($0 * $0 + $1 * $1).squareRoot() }
    var acceleration = [Double](repeating: .nan, count: n)
    for i in 1..<max(n, 1) { acceleration[i] = abs(speed[i] - speed[i - 1]) * fps }

    return WalkingFrameSignals(
        courtX: courtX, courtY: courtY, smoothedX: smoothedX, smoothedY: smoothedY,
        velocityX: velocityX, velocityY: velocityY, speed: speed, acceleration: acceleration,
        ankleSeparation: distance(leftAnkle, rightAnkle, over: bodyHeight),
        ankleVerticalGap: (0..<n).map { abs(leftAnkle.ys[$0] - rightAnkle.ys[$0]) / bodyHeight[$0] },
        hipAboveAnkle: (0..<n).map { (feet.ys[$0] - hip.ys[$0]) / bodyHeight[$0] },
        kneeAngle: nanMeanOfTwo(jointAngleDegrees(leftHip, leftKnee, leftAnkle),
                                jointAngleDegrees(rightHip, rightKnee, rightAnkle)),
        wristSpeedRelativeToHip: zip(limbSpeedRelativeToHip(limb: leftWrist, hip: hip, bodyHeightPx: bodyHeight, fps: fps),
                                     limbSpeedRelativeToHip(limb: rightWrist, hip: hip, bodyHeightPx: bodyHeight, fps: fps)).map(fmaxNaN),
        ankleSpeedRelativeToHip: zip(limbSpeedRelativeToHip(limb: leftAnkle, hip: hip, bodyHeightPx: bodyHeight, fps: fps),
                                     limbSpeedRelativeToHip(limb: rightAnkle, hip: hip, bodyHeightPx: bodyHeight, fps: fps)).map(fmaxNaN),
        valid: courtX.map { $0.isFinite ? 1 : 0 },
        onCourt: onCourt ?? [Double](repeating: .nan, count: n))
}

/// Mean, population std, p10/p50/p90 and max of the finite values in [first, last).
private func addStatistics(_ x: [Double], _ first: Int, _ last: Int, _ prefix: String,
                           into out: inout [String: Double], absolute: Bool = false) {
    let values = (first..<last).compactMap { i -> Double? in
        x[i].isFinite ? (absolute ? abs(x[i]) : x[i]) : nil
    }
    guard !values.isEmpty else {
        for name in ["mean", "std", "p10", "p50", "p90", "max"] { out["\(prefix)_\(name)"] = .nan }
        return
    }
    let average = mean(values)
    out["\(prefix)_mean"] = average
    out["\(prefix)_std"] = (values.reduce(0) { $0 + ($1 - average) * ($1 - average) } / Double(values.count)).squareRoot()
    out["\(prefix)_p10"] = percentile(values, 10)
    out["\(prefix)_p50"] = percentile(values, 50)
    out["\(prefix)_p90"] = percentile(values, 90)
    out["\(prefix)_max"] = values.max()!
}

/// walking.features._spectrum: (dominant in-band frequency, its share of
/// in-band power, in-band power per sample). numpy zero-pads the rfft to
/// max(64, length); only in-band bins are ever read, so only those are
/// computed, as a direct DFT.
private func cadenceSpectrum(_ signal: [Double], _ first: Int, _ last: Int, fps: Double) -> (Double, Double, Double) {
    var s = (first..<last).compactMap { signal[$0].isFinite ? signal[$0] : nil }
    let length = s.count
    guard length >= 8 else { return (.nan, .nan, .nan) }
    let average = mean(s)
    for i in 0..<length { s[i] -= average }
    let std = (s.reduce(0) { $0 + $1 * $1 } / Double(length)).squareRoot()
    guard std >= 1e-6 else { return (.nan, .nan, 0) }
    for i in 0..<length { s[i] *= 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(length - 1)) }  // np.hanning
    let fftLength = max(64, length)
    var powers: [Double] = [], frequencies: [Double] = []
    for k in 0...(fftLength / 2) {
        let frequency = Double(k) * fps / Double(fftLength)
        guard frequency >= WalkingFeatureRules.CADENCE_BAND.low && frequency <= WalkingFeatureRules.CADENCE_BAND.high else { continue }
        var real = 0.0, imaginary = 0.0
        for t in 0..<length {
            let angle = -2 * .pi * Double(k) * Double(t) / Double(fftLength)
            real += s[t] * cos(angle)
            imaginary += s[t] * sin(angle)
        }
        powers.append(real * real + imaginary * imaginary)
        frequencies.append(frequency)
    }
    let total = powers.reduce(0, +)
    guard !powers.isEmpty && total > 0 else { return (.nan, .nan, 0) }
    let strongest = powers.indices.max { powers[$0] < powers[$1] || (powers[$0] == powers[$1] && $0 > $1) }!
    return (frequencies[strongest], powers[strongest] / total, total / Double(length))
}

/// "w0.5", "w1", ... as Python's f"w{w:g}".
private func windowName(_ seconds: Double) -> String {
    seconds == seconds.rounded() ? "w\(Int(seconds))" : "w\(seconds)"
}

/// walking.features.window_features at the samples in `indices`. Rows are
/// float32, as Python returns them -- the tree thresholds were learned on float32.
public func walkingWindowFeatures(_ signals: WalkingFrameSignals, fps: Double,
                                  indices: [Int]) -> (rows: [[Float]], names: [String]) {
    let n = signals.speed.count
    let halfWidths = WalkingFeatureRules.WINDOWS_S.map { pythonRound($0 * fps / 2) }
    var rows: [[Float]] = []
    var names: [String] = []
    for t in indices {
        var features: [String: Double] = [:]
        var order: [String] = []
        func set(_ name: String, _ value: Double) {
            if features[name] == nil { order.append(name) }
            features[name] = value
        }
        set("court_x", signals.courtX[t])
        set("court_y", signals.courtY[t])
        for (windowIndex, seconds) in WalkingFeatureRules.WINDOWS_S.enumerated() {
            let half = halfWidths[windowIndex]
            let first = max(0, t - half), last = min(n, t + half + 1)
            let duration = Double(last - first) / fps
            let p = windowName(seconds)
            var stats: [String: Double] = [:]
            func statistics(_ x: [Double], _ name: String, absolute: Bool = false) {
                var block: [String: Double] = [:]
                addStatistics(x, first, last, "\(p)_\(name)", into: &block, absolute: absolute)
                for suffix in ["mean", "std", "p10", "p50", "p90", "max"] {
                    set("\(p)_\(name)_\(suffix)", block["\(p)_\(name)_\(suffix)"]!)
                }
                stats.merge(block) { $1 }
            }
            statistics(signals.speed, "speed")
            statistics(signals.velocityX, "vx", absolute: true)
            statistics(signals.velocityY, "vy", absolute: true)
            statistics(signals.acceleration, "acc")
            statistics(signals.ankleSeparation, "anksep")
            statistics(signals.kneeAngle, "knee")
            statistics(signals.wristSpeedRelativeToHip, "wrirel")
            statistics(signals.ankleSpeedRelativeToHip, "ankrel")
            statistics(signals.hipAboveAnkle, "hipank")

            var xs: [Double] = [], ys: [Double] = []
            for i in first..<last where signals.smoothedX[i].isFinite && signals.smoothedY[i].isFinite {
                xs.append(signals.smoothedX[i]); ys.append(signals.smoothedY[i])
            }
            if xs.count >= 2 {
                let net = ((xs.last! - xs.first!) * (xs.last! - xs.first!) + (ys.last! - ys.first!) * (ys.last! - ys.first!)).squareRoot()
                var path = 0.0
                for i in 1..<xs.count {
                    let dx = xs[i] - xs[i - 1], dy = ys[i] - ys[i - 1]
                    path += (dx * dx + dy * dy).squareRoot()
                }
                set("\(p)_net_disp", net)
                set("\(p)_net_speed", net / duration)
                set("\(p)_path_speed", path / duration)
                set("\(p)_straightness", net / max(path, 1e-6))
                set("\(p)_range_x", xs.max()! - xs.min()!)
                set("\(p)_range_y", ys.max()! - ys.min()!)
                set("\(p)_dy_signed", (ys.last! - ys.first!) / duration)
                set("\(p)_dx_abs", abs(xs.last! - xs.first!) / duration)
            } else {
                for name in ["net_disp", "net_speed", "path_speed", "straightness",
                             "range_x", "range_y", "dy_signed", "dx_abs"] { set("\(p)_\(name)", .nan) }
            }

            let speeds = (first..<last).compactMap { signals.speed[$0].isFinite ? signals.speed[$0] : nil }
            if speeds.isEmpty {
                set("\(p)_frac_slow", .nan); set("\(p)_frac_walkband", .nan); set("\(p)_frac_fast", .nan)
            } else {
                let count = Double(speeds.count)
                set("\(p)_frac_slow", Double(speeds.filter { $0 < 0.4 }.count) / count)
                set("\(p)_frac_walkband", Double(speeds.filter { $0 >= 0.4 && $0 < 2.5 }.count) / count)
                set("\(p)_frac_fast", Double(speeds.filter { $0 >= 2.5 }.count) / count)
            }
            let valid = finiteValues(signals.valid[first..<last])
            set("\(p)_valid", valid.isEmpty ? .nan : mean(valid))
            let onCourt = finiteValues(signals.onCourt[first..<last])
            set("\(p)_on_court", onCourt.isEmpty ? .nan : mean(onCourt))

            if seconds >= 1 {
                for (signal, tag) in [(signals.ankleSeparation, "anksep"), (signals.ankleVerticalGap, "ankdy"),
                                      (signals.hipAboveAnkle, "hipbob")] {
                    let (frequency, share, power) = cadenceSpectrum(signal, first, last, fps: fps)
                    set("\(p)_\(tag)_freq", frequency)
                    set("\(p)_\(tag)_share", share)
                    set("\(p)_\(tag)_power", power)
                }
            }
        }
        if names.isEmpty { names = order }
        rows.append(names.map { Float(features[$0]!) })
    }
    return (rows, names)
}

/// The walking classifier's gradient-boosted trees, evaluated from JSON.
public struct WalkingModel: Decodable {
    struct Tree: Decodable {
        let is_leaf: [Int]
        let feature: [Int]
        let threshold: [Double]
        let missing_go_left: [Int]
        let left: [Int]
        let right: [Int]
        let value: [Double]

        /// Walk from the root: `x <= threshold` goes left, NaN follows the
        /// split's missing-value direction (sklearn's raw-feature predict).
        func leafValue(_ features: [Float]) -> Double {
            var node = 0
            while is_leaf[node] == 0 {
                let x = Double(features[feature[node]])
                if x.isNaN {
                    node = missing_go_left[node] != 0 ? left[node] : right[node]
                } else {
                    node = x <= threshold[node] ? left[node] : right[node]
                }
            }
            return value[node]
        }
    }

    public let feature_names: [String]
    public let post_stride: Int
    public let baseline: Double
    let trees: [Tree]

    /// The model shipped with the package.
    public static func bundled() throws -> WalkingModel {
        let url = Bundle.module.url(forResource: "walking_model", withExtension: "json")!
        return try JSONDecoder().decode(WalkingModel.self, from: Data(contentsOf: url))
    }

    public func walkingProbability(_ features: [Float]) -> Double {
        let raw = trees.reduce(baseline) { $0 + $1.leafValue(features) }
        return 1 / (1 + exp(-raw))
    }
}

/// walking.predict.predict_video's probability trace for one near-player track.
/// `poseStride` is the pose pass's stride in SOURCE frames; the model's post
/// stride is in source frames too, so the decision stride is their ratio.
public func walkingProbability(model: WalkingModel, signals: WalkingFrameSignals,
                               fps: Double, poseStride: Int) -> [Double] {
    let n = signals.speed.count
    let stride = max(1, pythonRound(Double(model.post_stride) / Double(poseStride)))
    let indices = Array(Swift.stride(from: 0, to: n, by: stride))
    let (rows, names) = walkingWindowFeatures(signals, fps: fps, indices: indices)
    precondition(rows.isEmpty || names == model.feature_names,
                 "walking feature layout changed since the model was trained")
    var probability = [Double](repeating: 0, count: n)
    for (row, t) in zip(rows, indices) {
        let p = model.walkingProbability(row)
        for i in t..<min(t + stride, n) { probability[i] = p }
    }
    return probability
}
