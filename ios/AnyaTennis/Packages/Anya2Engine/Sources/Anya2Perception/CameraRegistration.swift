// CameraRegistration.swift  (camera.py: register, _fit, _sane, _register_via_last_good)
//
// For a frame t, the warp W_t that maps it back onto the calibration frame,
// so court projections survive the phone being bumped:
//     court metres = H_ref * W_t * (image point at frame t)
// Each sample is registered DIRECTLY against the calibration frame (never
// chained), by ORB features + ratio-tested matches + a RANSAC homography,
// refined on the court-surface matches alone when those agree. A failed
// sample tries once more through the last sample that did register.
//
// All images are greyscale at the analysis size, 960x540.
import Anya2Engine
import Anya2OpenCV
import Foundation

public enum CameraRules {
    public static let SAMPLE_FPS = 5.0          // samples per second; a jostle is a step, not a signal
    public static let ORB_FEATURES = 4000       // high, so the low-texture court still gets features
    public static let LOWE_RATIO = 0.75         // ratio test for a good match
    public static let RANSAC_PX = 3.0
    public static let MIN_INLIERS = 30          // fewer than this and the fit is not evidence
    // The ground-plane re-fit replaces the global fit only with at least this
    // many inliers AND court corners within GROUND_AGREE_PX of the global fit's.
    public static let MIN_GROUND_INLIERS = 60
    public static let GROUND_AGREE_PX = 4.0
    public static let MAX_SHIFT_PX = 240.0      // no frame corner may move more than this
    public static let AREA_LO = 0.70, AREA_HI = 1.45   // warped frame area / original area
    public static let SMOOTH_SAMPLES = 5        // running-median width (1 s at 5 Hz)
    public static let COURT_MASK_PAD_PX = 25    // the court lines sit at the quad's edge

    /// Source frames between camera samples (camera.estimate).
    public static func sampleStep(sourceFPS: Double) -> Int { max(1, pythonRound(sourceFPS / SAMPLE_FPS)) }
}

/// A greyscale frame, borrowed (not copied) for the length of one call.
public struct GrayFrame {
    public let pixels: UnsafePointer<UInt8>
    public let width: Int, height: Int, bytesPerRow: Int

    public init(pixels: UnsafePointer<UInt8>, width: Int, height: Int, bytesPerRow: Int) {
        self.pixels = pixels; self.width = width; self.height = height; self.bytesPerRow = bytesPerRow
    }
}

/// ORB features of one frame (camera._detect).
public struct Features {
    public let points: [Float]          // [count, 2]
    public let descriptors: Data?       // [count, 32]; nil = too few to use
    public var count: Int { points.count / 2 }

    static let none = Features(points: [], descriptors: nil)

    init(points: [Float], descriptors: Data?) { self.points = points; self.descriptors = descriptors }

    public init(of frame: GrayFrame) {
        guard let found = A2OpenCV.orbFeatures(ofGray: frame.pixels, width: frame.width, height: frame.height,
                                               bytesPerRow: frame.bytesPerRow, maxFeatures: CameraRules.ORB_FEATURES),
              found.count >= 8
        else { self = .none; return }
        self.init(points: found.points.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw.bindMemory(to: Float.self)) },
                  descriptors: found.descriptors)
    }
}

/// One registration's answer.
public struct Registration {
    public let warp: Homography?    // this frame -> calibration frame; nil = failed
    public let inliers: Int
    public let features: Features
}

/// The calibration frame, and everything registering against it needs.
public final class CameraReference {
    public let courtCorners: [(x: Double, y: Double)]   // tapped, analysis px
    let features: Features
    private let courtMask: [UInt8]
    private let width: Int, height: Int

    public init(calibrationFrame: GrayFrame, courtCorners: [(x: Double, y: Double)]) throws {
        guard courtCorners.count == 4,
              courtCorners.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 })
        else { throw CameraError.badCourtCorners }
        self.courtCorners = courtCorners
        width = calibrationFrame.width; height = calibrationFrame.height
        features = Features(of: calibrationFrame)
        guard features.descriptors != nil else { throw CameraError.noFeaturesInCalibrationFrame }
        // camera._court_mask: corners truncated to int32, as numpy's astype does.
        var corners = courtCorners.flatMap { [Int32($0.x), Int32($0.y)] }
        let mask = A2OpenCV.courtMask(withCorners: Data(bytes: &corners, count: corners.count * 4),
                                      width: width, height: height, padPixels: CameraRules.COURT_MASK_PAD_PX)
        courtMask = [UInt8](mask)
        guard courtMask.count == width * height else { throw CameraError.badCourtCorners }
    }

    /// Which calibration-frame points fall on the (dilated) court surface.
    func isOnCourt(x: Float, y: Float) -> Bool {
        let column = min(max(Int(Double(x).rounded(.toNearestOrEven)), 0), width - 1)
        let row = min(max(Int(Double(y).rounded(.toNearestOrEven)), 0), height - 1)
        return courtMask[row * width + column] != 0
    }

    /// camera.register: warp `frame` onto the calibration frame. The
    /// approximate match is retried with brute force if it fails.
    public func register(_ frame: GrayFrame) -> Registration {
        let current = Features(of: frame)
        for approximate in [true, false] {
            let (source, destination) = ratioMatches(current, features, approximate: approximate)
            let fit = fitWarp(source, destination, onCourt: { self.isOnCourt(x: $0, y: $1) },
                              probe: courtCorners)
            if fit.warp != nil { return Registration(warp: fit.warp, inliers: fit.inliers, features: current) }
        }
        return Registration(warp: nil, inliers: 0, features: current)
    }

    /// camera._register_via_last_good: match against the last sample that
    /// registered and compose with its warp. One link only.
    public func registerViaLastGood(_ current: Features, lastGoodWarp: Homography,
                                    lastGoodFeatures: Features) -> Registration {
        for approximate in [true, false] {
            let (source, destination) = ratioMatches(current, lastGoodFeatures, approximate: approximate)
            let link = fitWarp(source, destination, onCourt: { _, _ in true }, probe: nil)
            if let warp = link.warp {
                let composed = lastGoodWarp * warp
                return isSane(composed) ? Registration(warp: composed, inliers: link.inliers, features: current)
                                        : Registration(warp: nil, inliers: 0, features: current)
            }
        }
        return Registration(warp: nil, inliers: 0, features: current)
    }
}

public enum CameraError: Error { case noFeaturesInCalibrationFrame, badCourtCorners }

/// camera._match: ratio-tested correspondences as float32 point pairs.
func ratioMatches(_ current: Features, _ reference: Features, approximate: Bool) -> (source: [Float], destination: [Float]) {
    guard let query = current.descriptors, let train = reference.descriptors,
          current.count >= 2, reference.count >= 2 else { return ([], []) }
    let pairs = A2OpenCV.ratioMatches(ofQuery: query, train: train, approximate: approximate,
                                      ratio: CameraRules.LOWE_RATIO)
    var source: [Float] = [], destination: [Float] = []
    pairs.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let indices = raw.bindMemory(to: Int32.self)
        for i in Swift.stride(from: 0, to: indices.count, by: 2) {
            let q = Int(indices[i]), t = Int(indices[i + 1])
            source += [current.points[q * 2], current.points[q * 2 + 1]]
            destination += [reference.points[t * 2], reference.points[t * 2 + 1]]
        }
    }
    return (source, destination)
}

/// camera._fit: global RANSAC homography, then the ground-plane re-fit if it
/// earns it. `probe` (the court corners) is where the two fits are compared;
/// without it the global fit is returned as is.
func fitWarp(_ source: [Float], _ destination: [Float], onCourt: (Float, Float) -> Bool,
             probe: [(x: Double, y: Double)]?) -> (warp: Homography?, inliers: Int) {
    func ransac(_ s: [Float], _ d: [Float]) -> (Homography, [Bool])? {
        guard let fit = A2OpenCV.ransacHomography(from: Data(bytes: s, count: s.count * 4),
                                                  to: Data(bytes: d, count: d.count * 4),
                                                  threshold: CameraRules.RANSAC_PX, maxIterations: 5000,
                                                  confidence: 0.999) else { return nil }
        let matrix = fit.matrix.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw.bindMemory(to: Double.self)) }
        return (Homography(matrix), fit.inliers.map { $0 != 0 })
    }
    let pairs = source.count / 2
    guard pairs >= CameraRules.MIN_INLIERS, let (global, inliers) = ransac(source, destination) else { return (nil, 0) }
    let globalCount = inliers.filter { $0 }.count
    guard globalCount >= CameraRules.MIN_INLIERS, isSane(global) else { return (nil, 0) }
    guard let probe else { return (global, globalCount) }

    let groundPairs = (0..<pairs).filter { inliers[$0] && onCourt(destination[$0 * 2], destination[$0 * 2 + 1]) }
    guard groundPairs.count >= CameraRules.MIN_GROUND_INLIERS else { return (global, globalCount) }
    guard let (ground, groundInliers) = ransac(groundPairs.flatMap { [source[$0 * 2], source[$0 * 2 + 1]] },
                                               groundPairs.flatMap { [destination[$0 * 2], destination[$0 * 2 + 1]] }),
          isSane(ground) else { return (global, globalCount) }
    let groundCount = groundInliers.filter { $0 }.count
    guard groundCount >= CameraRules.MIN_GROUND_INLIERS else { return (global, globalCount) }
    // Agreement gate: the re-fit may refine the global fit, not replace it.
    let disagreement = probe.map { corner -> Double in
        let a = global.apply(x: corner.x, y: corner.y), b = ground.apply(x: corner.x, y: corner.y)
        return hypot(a.x - b.x, a.y - b.y)
    }.max()!
    return disagreement > CameraRules.GROUND_AGREE_PX ? (global, globalCount) : (ground, groundCount)
}

let analysisFrameCorners: [(x: Double, y: Double)] = [
    (0, 0), (Double(Court.ANALYSIS_WIDTH), 0),
    (Double(Court.ANALYSIS_WIDTH), Double(Court.ANALYSIS_HEIGHT)), (0, Double(Court.ANALYSIS_HEIGHT)),
]

/// camera._sane: reject a warp that folds, collapses or flings the frame.
/// Not a quality score -- a wrong fit is wildly wrong, and this only has to catch that.
func isSane(_ warp: Homography) -> Bool {
    guard warp.m.allSatisfy(\.isFinite) else { return false }
    let warped = analysisFrameCorners.map { warp.apply(x: $0.x, y: $0.y) }
    guard warped.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return false }
    for (moved, corner) in zip(warped, analysisFrameCorners)
    where max(abs(moved.x - corner.x), abs(moved.y - corner.y)) > CameraRules.MAX_SHIFT_PX { return false }
    // Signed (shoelace) area: negative means the quad was folded over.
    var twiceArea = 0.0
    for i in 0..<4 {
        let a = warped[i], b = warped[(i + 1) % 4]
        twiceArea += a.x * b.y - a.y * b.x
    }
    let ratio = 0.5 * twiceArea / Double(Court.ANALYSIS_WIDTH * Court.ANALYSIS_HEIGHT)
    return ratio >= CameraRules.AREA_LO && ratio <= CameraRules.AREA_HI
}
