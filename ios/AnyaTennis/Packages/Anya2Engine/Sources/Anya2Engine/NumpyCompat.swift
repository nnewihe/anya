// NumpyCompat.swift
//
// The handful of numpy behaviours the Python pipeline relies on, reproduced
// exactly. Where Swift's obvious equivalent differs from numpy, the numpy
// behaviour wins, because the Python pipeline is the reference these results
// are checked against.
//
// The three that matter most:
//   - Python's round() is round-half-to-even. Window sizes are computed as
//     int(round(seconds * fps)) and land on .5 at 15 Hz (0.7 s * 15 = 10.5
//     -> 10), so use `pythonRound`, never `.rounded()`.
//   - np.fmax / np.fmin IGNORE NaN; np.maximum / np.clip PROPAGATE it.
//   - np.convolve(..., mode="same") centres an even-width window one sample
//     to the left: output i sums x[i - w/2 ... i + (w-1)/2].
import Foundation

/// Python's `round(x)` for a float: halves go to the even neighbour.
public func pythonRound(_ x: Double) -> Int {
    let floorValue = x.rounded(.down)
    let fraction = x - floorValue
    if fraction > 0.5 { return Int(floorValue) + 1 }
    if fraction < 0.5 { return Int(floorValue) }
    let lower = Int(floorValue)
    return lower % 2 == 0 ? lower : lower + 1
}

/// Python's `round(x, digits)`, for the reporting fields that use it
/// (scores and timestamps written into events; never fed back into a decision).
public func pythonRound(_ x: Double, digits: Int) -> Double {
    guard x.isFinite else { return x }
    let scale = pow(10.0, Double(digits))
    return Double(pythonRound(x * scale)) / scale
}

/// `np.fmax`: NaN is ignored unless both are NaN.
@inline(__always) public func fmaxNaN(_ a: Double, _ b: Double) -> Double {
    if a.isNaN { return b }
    if b.isNaN { return a }
    return a > b ? a : b
}

/// `np.fmin`: NaN is ignored unless both are NaN.
@inline(__always) public func fminNaN(_ a: Double, _ b: Double) -> Double {
    if a.isNaN { return b }
    if b.isNaN { return a }
    return a < b ? a : b
}

/// `np.clip` on one value: NaN in, NaN out.
@inline(__always) public func clip(_ x: Double, _ low: Double, _ high: Double) -> Double {
    if x.isNaN { return x }
    return x < low ? low : (x > high ? high : x)
}

/// `np.convolve(x, np.ones(width), mode="same")`: a centred running sum that
/// is clipped at the array ends. NaN propagates, as in numpy.
public func convolveSameOnes(_ x: [Double], width: Int) -> [Double] {
    let n = x.count
    let before = width / 2, after = (width - 1) / 2
    var out = [Double](repeating: 0, count: n)
    for i in 0..<n {
        var total = 0.0
        let first = max(0, i - before), last = min(n - 1, i + after)
        if first <= last { for j in first...last { total += x[j] } }
        out[i] = total
    }
    return out
}

/// `np.convolve(x, np.ones(width) / width, mode="same")`: a centred mean that
/// keeps dividing by `width` at the ends, exactly as numpy does.
public func convolveSameMean(_ x: [Double], width: Int) -> [Double] {
    let n = x.count
    let before = width / 2, after = (width - 1) / 2
    let weight = 1.0 / Double(width)
    var out = [Double](repeating: 0, count: n)
    for i in 0..<n {
        var total = 0.0
        let first = max(0, i - before), last = min(n - 1, i + after)
        if first <= last { for j in first...last { total += x[j] * weight } }
        out[i] = total
    }
    return out
}

/// `np.percentile(values, q)` with numpy's default linear method.
/// `values` must be NaN-free; an empty array gives NaN.
public func percentile(_ values: [Double], _ q: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    let sorted = values.sorted()
    let position = q / 100.0 * Double(sorted.count - 1)
    let lowerIndex = Int(position.rounded(.down)), upperIndex = Int(position.rounded(.up))
    let fraction = position - Double(lowerIndex)
    let a = sorted[lowerIndex], b = sorted[upperIndex]
    // numpy's _lerp switches formula at 0.5 so ties land identically.
    return fraction >= 0.5 ? b - (b - a) * (1 - fraction) : a + (b - a) * fraction
}

public func median(_ values: [Double]) -> Double { percentile(values, 50) }

public func mean(_ values: [Double]) -> Double {
    values.reduce(0, +) / Double(values.count)
}

/// The finite values of a sequence, in order.
public func finiteValues<S: Sequence>(_ values: S) -> [Double] where S.Element == Double {
    values.filter { $0.isFinite }
}

/// A stable sort (Python's `sorted`): ties keep their input order.
public func stableSorted<T>(_ items: [T], by areInIncreasingOrder: (T, T) -> Bool) -> [T] {
    items.enumerated()
        .sorted { lhs, rhs in
            if areInIncreasingOrder(lhs.element, rhs.element) { return true }
            if areInIncreasingOrder(rhs.element, lhs.element) { return false }
            return lhs.offset < rhs.offset
        }
        .map(\.element)
}

/// Round to float32 precision, for places where Python does arithmetic on
/// float32 array elements (the stored tracks are float32).
@inline(__always) public func toFloat32Precision(_ x: Double) -> Double { Double(Float(x)) }
