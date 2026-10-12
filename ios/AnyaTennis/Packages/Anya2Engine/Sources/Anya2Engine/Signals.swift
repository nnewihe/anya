// Signals.swift  --  port of pipeline/anya2/signals.py
//
// Small, parameter-free maths shared by the detectors: ramps, running maxima,
// run-length decomposition, gap filling, keypoint reads. Thresholds never
// live here; each detector owns its own.
//
// TWO FAMILIES OF POSE HELPERS -- know which one you are using.
// They look alike but do NOT behave alike; swapping them changes results.
//
//   anya2's own                         used by far serve, point end
//     fillInteriorGaps        each column on its own; never fills a gap
//                             touching the first or last sample
//     keypointXY              NaN where confidence <= confFloor
//     midpointBothRequired    NaN unless BOTH points are present
//
//   the walking classifier's           used by near serve, near_end
//     fillGapsBetweenSeen     fills between samples where column 0 was seen
//     keypointXYWalking       NaN where confidence < 0.3 (walking KP_CONF)
//     midpointWithFallback    falls back to whichever point exists
//     nanSkippingMovingMean   centred moving mean that skips NaN
//     jointAngleDegrees       interior angle at the middle joint
import Foundation

// MARK: - COCO keypoint indices

public enum Keypoint {
    public static let nose = 0
    public static let leftEye = 1, rightEye = 2
    public static let leftEar = 3, rightEar = 4
    public static let leftShoulder = 5, rightShoulder = 6
    public static let leftElbow = 7, rightElbow = 8
    public static let leftWrist = 9, rightWrist = 10
    public static let leftHip = 11, rightHip = 12
    public static let leftKnee = 13, rightKnee = 14
    public static let leftAnkle = 15, rightAnkle = 16
    public static let count = 17
}

// MARK: - Data shapes

/// A per-sample 2-D point track: `xs[i]`, `ys[i]`; NaN where not seen.
public struct PointTrack {
    public var xs: [Double]
    public var ys: [Double]
    public init(xs: [Double], ys: [Double]) { self.xs = xs; self.ys = ys }
    public init(missingCount n: Int) {
        xs = [Double](repeating: .nan, count: n)
        ys = [Double](repeating: .nan, count: n)
    }
    public var count: Int { xs.count }
}

/// A per-sample table with a fixed number of columns, row-major.
/// One player's keypoints are `columns = 51` (17 x (x, y, confidence));
/// one player's boxes are `columns = 4` (x1, y1, x2, y2).
public struct SampleTable {
    public let rows: Int
    public let columns: Int
    public var values: [Double]
    public init(rows: Int, columns: Int, values: [Double]) {
        precondition(values.count == rows * columns)
        self.rows = rows; self.columns = columns; self.values = values
    }
    @inline(__always) public subscript(row: Int, column: Int) -> Double {
        get { values[row * columns + column] }
        set { values[row * columns + column] = newValue }
    }
    public func column(_ c: Int) -> [Double] { (0..<rows).map { self[$0, c] } }
}

// MARK: - Scores and windows

/// Linear 0 -> 1 as x goes from `low` to `high` (either order), clipped.
/// NaN in, NaN out. Pass high < low for a falling ramp.
@inline(__always) public func ramp(_ x: Double, _ low: Double, _ high: Double) -> Double {
    clip((x - low) / (high - low), 0, 1)
}

public func ramp(_ xs: [Double], _ low: Double, _ high: Double) -> [Double] {
    xs.map { ramp($0, low, high) }
}

/// Centred running max over `width` samples; NaN is ignored (acts as -inf).
/// Near the ends the window just gets shorter.
/// See NOTES: "The trophy is a phase, not an instant".
public func runningMax(_ x: [Double], width: Int) -> [Double] {
    guard width > 1 else { return x }
    let reach = width / 2, n = x.count
    var out = x
    for i in 0..<n {
        var best = x[i]
        for j in max(0, i - reach)...min(n - 1, i + reach) { best = fmaxNaN(best, x[j]) }
        out[i] = best
    }
    return out
}

/// The [start, stop) index ranges where `mask` is true.
public func trueRuns(_ mask: [Bool]) -> [(start: Int, stop: Int)] {
    var runs: [(start: Int, stop: Int)] = []
    var i = 0
    while i < mask.count {
        guard mask[i] else { i += 1; continue }
        let start = i
        while i < mask.count && mask[i] { i += 1 }
        runs.append((start, i))
    }
    return runs
}

/// Thin events so no two kept events are within `gap` seconds of each other.
/// `strongest`: keep the highest-`p` member of each cluster (ties: earliest),
/// right when a cluster is one action detected several times. Otherwise keep
/// the EARLIEST member, right when the first one is the one being timed.
public func keepOnePerWindow<T>(_ events: [T], gap: Double, strongest: Bool = true,
                                time: (T) -> Double, score: (T) -> Double) -> [T] {
    if strongest {
        let byStrength = stableSorted(events) {
            score($0) != score($1) ? score($0) > score($1) : time($0) < time($1)
        }
        var kept: [T] = []
        for event in byStrength where kept.allSatisfy({ abs(time(event) - time($0)) >= gap }) {
            kept.append(event)
        }
        return stableSorted(kept) { time($0) < time($1) }
    }
    var kept: [T] = []
    for event in stableSorted(events, by: { time($0) < time($1) }) {
        if let last = kept.last, time(event) - time(last) < gap { continue }
        kept.append(event)
    }
    return kept
}

/// Centred moving mean over `width` samples that ignores NaN; NaN only where
/// the whole window is NaN. Same formula as the walking classifier's _movmean.
public func nanSkippingMovingMean(_ x: [Double], width: Int) -> [Double] {
    guard width > 1 else { return x }
    let total = convolveSameOnes(x.map { $0.isFinite ? $0 : 0 }, width: width)
    let count = convolveSameOnes(x.map { $0.isFinite ? 1 : 0 }, width: width)
    return zip(total, count).map { $1 > 0 ? $0 / max($1, 1e-9) : .nan }
}

/// True from the first sample at or above `enterAt` until a sample drops below `leaveAt`.
public func hysteresis(_ x: [Double], enterAt: Double, leaveAt: Double) -> [Bool] {
    var on = false
    return x.map { value in
        on = on ? value >= leaveAt : value >= enterAt
        return on
    }
}

/// Speed of a point track in units per second; element 0 is NaN.
public func frameToFrameSpeed(_ points: PointTrack, fps: Double) -> [Double] {
    var speed = [Double](repeating: .nan, count: points.count)
    for i in 1..<max(points.count, 1) {
        let dx = points.xs[i] - points.xs[i - 1], dy = points.ys[i] - points.ys[i - 1]
        speed[i] = (dx * dx + dy * dy).squareRoot() * fps
    }
    return speed
}

/// How fast a joint moves RELATIVE TO THE HIP, in body heights per second.
/// Relative to the hip so that running across court is not arm activity,
/// while a racket take-back while standing still is.
public func limbSpeedRelativeToHip(limb: PointTrack, hip: PointTrack,
                                   bodyHeightPx: [Double], fps: Double) -> [Double] {
    let relative = PointTrack(
        xs: (0..<limb.count).map { (limb.xs[$0] - hip.xs[$0]) / bodyHeightPx[$0] },
        ys: (0..<limb.count).map { (limb.ys[$0] - hip.ys[$0]) / bodyHeightPx[$0] })
    return frameToFrameSpeed(relative, fps: fps)
}

/// How fast a player's box moves across the image, in body heights per second,
/// measured on the bottom-centre x and the bottom y.
public func boxTravelSpeed(boxes: SampleTable, bodyHeightPx: [Double], fps: Double) -> [Double] {
    var speed = [Double](repeating: .nan, count: boxes.rows)
    for i in 1..<max(boxes.rows, 1) {
        let dx = 0.5 * (boxes[i, 0] + boxes[i, 2]) - 0.5 * (boxes[i - 1, 0] + boxes[i - 1, 2])
        let dy = boxes[i, 3] - boxes[i - 1, 3]
        speed[i] = (dx * dx + dy * dy).squareRoot() / bodyHeightPx[i] * fps
    }
    return speed
}

/// Box heights (y2 - y1) of a box table.
public func boxHeights(_ boxes: SampleTable) -> [Double] {
    (0..<boxes.rows).map { boxes[$0, 3] - boxes[$0, 1] }
}

// MARK: - anya2's own pose helpers

/// Linearly fill runs of NaN no longer than `maxGap` samples, each column on
/// its own. Gaps touching the first or last sample, and longer gaps, are left
/// alone: a filled long hole would be a made-up posture.
public func fillInteriorGaps(_ table: SampleTable, maxGap: Int) -> SampleTable {
    var out = table
    let n = table.rows
    for c in 0..<table.columns {
        let seen = (0..<n).map { table[$0, c].isFinite }
        guard seen.filter({ $0 }).count >= 2 else { continue }
        for run in trueRuns(seen.map { !$0 }) where run.stop - run.start <= maxGap
            && run.start > 0 && run.stop < n {
            // np.interp between the two seen neighbours.
            let x0 = Double(run.start - 1), x1 = Double(run.stop)
            let y0 = table[run.start - 1, c], y1 = table[run.stop, c]
            let slope = (y1 - y0) / (x1 - x0)
            for i in run.start..<run.stop { out[i, c] = slope * (Double(i) - x0) + y0 }
        }
    }
    return out
}

/// Image position of keypoint `index`; NaN where confidence <= `confFloor`.
public func keypointXY(_ keypoints: SampleTable, _ index: Int, confFloor: Double = 0) -> PointTrack {
    var out = PointTrack(missingCount: keypoints.rows)
    for i in 0..<keypoints.rows where !(keypoints[i, index * 3 + 2] <= confFloor) {
        out.xs[i] = keypoints[i, index * 3]
        out.ys[i] = keypoints[i, index * 3 + 1]
    }
    return out
}

/// Midpoint of two tracks; NaN wherever either is missing.
public func midpointBothRequired(_ a: PointTrack, _ b: PointTrack) -> PointTrack {
    PointTrack(xs: zip(a.xs, b.xs).map { 0.5 * ($0 + $1) },
               ys: zip(a.ys, b.ys).map { 0.5 * ($0 + $1) })
}

// MARK: - The walking classifier's pose helpers (walking/features.py)

/// The walking classifier's keypoint confidence floor (walking.features.KP_CONF).
public let WALKING_KP_CONF = 0.3

/// Linearly fill gaps of at most `maxGap` samples between rows where COLUMN 0
/// was seen, in every column from that pair of rows (walking._interp_gaps).
public func fillGapsBetweenSeen(_ table: SampleTable, maxGap: Int) -> SampleTable {
    var out = table
    let seenRows = (0..<table.rows).filter { !table[$0, 0].isNaN }
    guard seenRows.count >= 2 else { return out }
    for (a, b) in zip(seenRows, seenRows.dropFirst()) {
        let gap = b - a - 1
        guard gap > 0 && gap <= maxGap else { continue }
        for c in 0..<table.columns {
            let ya = table[a, c], yb = table[b, c]
            let slope = (yb - ya) / Double(b - a)
            for i in (a + 1)..<b { out[i, c] = slope * Double(i - a) + ya }
        }
    }
    return out
}

/// Keypoint position, NaN where confidence < 0.3 -- strictly below
/// (walking._kp_xy).
public func keypointXYWalking(_ keypoints: SampleTable, _ index: Int) -> PointTrack {
    var out = PointTrack(missingCount: keypoints.rows)
    for i in 0..<keypoints.rows where !(keypoints[i, index * 3 + 2] < WALKING_KP_CONF) {
        out.xs[i] = keypoints[i, index * 3]
        out.ys[i] = keypoints[i, index * 3 + 1]
    }
    return out
}

/// Midpoint that falls back to whichever point exists, judged on x
/// (walking._mid).
public func midpointWithFallback(_ a: PointTrack, _ b: PointTrack) -> PointTrack {
    var out = midpointBothRequired(a, b)
    for i in 0..<out.count where out.xs[i].isNaN {
        if !a.xs[i].isNaN { out.xs[i] = a.xs[i]; out.ys[i] = a.ys[i] }
        else if !b.xs[i].isNaN { out.xs[i] = b.xs[i]; out.ys[i] = b.ys[i] }
    }
    return out
}

/// Interior angle at `middle` of the chain a-middle-c, degrees; NaN if any
/// point is missing (walking._angle).
public func jointAngleDegrees(_ a: PointTrack, _ middle: PointTrack, _ c: PointTrack) -> [Double] {
    (0..<a.count).map { i in
        let v1x = a.xs[i] - middle.xs[i], v1y = a.ys[i] - middle.ys[i]
        let v2x = c.xs[i] - middle.xs[i], v2y = c.ys[i] - middle.ys[i]
        let norms = (v1x * v1x + v1y * v1y).squareRoot() * (v2x * v2x + v2y * v2y).squareRoot()
        let cosine = clip((v1x * v2x + v1y * v2y) / max(norms, 1e-9), -1, 1)
        return cosine.isNaN ? .nan : acos(cosine) * 180 / .pi
    }
}

/// Element-wise `np.nanmean` of two arrays: NaN only where both are NaN.
public func nanMeanOfTwo(_ a: [Double], _ b: [Double]) -> [Double] {
    zip(a, b).map { x, y in
        switch (x.isFinite, y.isFinite) {
        case (true, true): return (x + y) / 2
        case (true, false): return x
        case (false, true): return y
        default: return .nan
        }
    }
}

/// Distance between two point tracks divided by a per-sample scale; NaN propagates.
public func distance(_ a: PointTrack, _ b: PointTrack, over scale: [Double]) -> [Double] {
    (0..<a.count).map { i in
        let dx = a.xs[i] - b.xs[i], dy = a.ys[i] - b.ys[i]
        return (dx * dx + dy * dy).squareRoot() / scale[i]
    }
}
