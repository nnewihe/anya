// LiveCameraTrack.swift  (camera.estimate's loop, _fill_failed_samples, smooth)
//
// The camera track built while recording, one sample at a time, with every
// committed warp EQUAL to what the batch rules give for the whole recording:
//   fill    a failed sample takes the previous good warp; failures before
//           the first good one take the FIRST good warp -- so nothing is
//           committed until some sample has registered
//   smooth  the running median of 5 over where 8 points land (frame corners
//           + court corners), re-fitted to a warp. Sample i's window is
//           i-2...i+2, so it is final once sample i+2 exists; only the last
//           two samples (whose window clips at the end) wait for Stop.
import Anya2Engine
import Anya2OpenCV
import Foundation

public final class LiveCameraTrack {
    public let reference: CameraReference
    public let sampleStep: Int

    private var frames: [Int] = []
    private var registered: [Homography?] = []         // per sample, nil = failed
    private var filled: [Homography] = []               // after the fill rule
    private var lastGood: (warp: Homography, features: Features)?
    /// The smoothed, final warps so far (a prefix of the batch track).
    public private(set) var committed: [Homography] = []

    public init(reference: CameraReference, sampleStep: Int) {
        self.reference = reference; self.sampleStep = sampleStep
    }

    /// Source frames that are camera samples.
    public func isSampleFrame(_ sourceFrame: Int) -> Bool { sourceFrame % sampleStep == 0 }

    /// The source frame of the latest committed sample, or nil.
    public var committedThroughFrame: Int? { committed.isEmpty ? nil : frames[committed.count - 1] }

    /// The most recent sample's own registration (nil = it failed), before
    /// fill and smoothing: what the perception log records.
    public var lastRegisteredWarp: Homography? { registered.last ?? nil }

    /// The committed part of the track, for Geometry.
    public var committedTrack: CameraTrack? {
        committed.isEmpty ? nil : CameraTrack(frames: Array(frames.prefix(committed.count)), warps: committed)
    }

    /// Register one sample frame (every `sampleStep`-th source frame, in order).
    public func add(_ frame: GrayFrame, sourceFrame: Int) {
        var registration = reference.register(frame)
        if registration.warp == nil, let lastGood {
            registration = reference.registerViaLastGood(registration.features, lastGoodWarp: lastGood.warp,
                                                         lastGoodFeatures: lastGood.features)
        }
        add(warp: registration.warp, sourceFrame: sourceFrame)
        if let warp = registration.warp { lastGood = (warp, registration.features) }
    }

    /// The bookkeeping half of `add`, separate so tests can feed warps directly.
    func add(warp: Homography?, sourceFrame: Int) {
        frames.append(sourceFrame)
        registered.append(warp)
        fill()
        commitSmoothed(final: false)
    }

    /// The whole track at Stop: the last samples' clipped windows, or the
    /// unsmoothed warps when there are fewer than 5 samples (as the batch does).
    public func finish() -> CameraTrack {
        if filled.count < registered.count {   // nothing ever registered
            filled = Array(repeating: .identity, count: registered.count)
        }
        commitSmoothed(final: true)
        return frames.isEmpty ? .identity : CameraTrack(frames: frames, warps: committed)
    }

    /// Extend `filled` as far as the fill rule allows.
    private func fill() {
        guard let firstGood = registered.firstIndex(where: { $0 != nil }) else { return }
        while filled.count < registered.count {
            let i = filled.count
            if let warp = registered[i] { filled.append(warp) }
            else { filled.append(i < firstGood ? registered[firstGood]! : filled[i - 1]) }
        }
    }

    private func commitSmoothed(final: Bool) {
        let n = filled.count, k = CameraRules.SMOOTH_SAMPLES, half = k / 2
        if final && n < k {
            committed = filled
            return
        }
        // Without Stop, sample i is final once i + half < n (and n >= k, so
        // its window is not clipped by the start AND the end at once).
        let limit = final ? n : (n >= k ? n - half : 0)
        while committed.count < limit {
            let i = committed.count
            let start = min(max(i - half, 0), n - k)
            committed.append(smoothedWarp(window: start..<(start + k), sample: i))
        }
    }

    /// camera.smooth for one sample: median positions of the 8 points over the
    /// window, least-squares re-fit; keep the raw warp if that is not sane.
    private func smoothedWarp(window: Range<Int>, sample: Int) -> Homography {
        let points = analysisFrameCorners + reference.courtCorners
        let located = window.map { j -> [(x: Double, y: Double)] in
            let inverse = filled[j].inverse
            return points.map { inverse.apply(x: $0.x, y: $0.y) }
        }
        var median: [Double] = []
        for p in 0..<points.count {
            median.append(Anya2Engine.median(located.map { $0[p].x }))
            median.append(Anya2Engine.median(located.map { $0[p].y }))
        }
        guard let warp = smoothingFit(source: median, target: points.flatMap { [$0.x, $0.y] }) else { return filled[sample] }
        return isSane(warp) ? warp : filled[sample]
    }
}

/// cv2.findHomography(source, target, 0): plain least squares over point
/// pairs (flattened x, y), as camera.smooth re-fits from the medians.
func smoothingFit(source: [Double], target: [Double]) -> Homography? {
    guard let matrix = A2OpenCV.leastSquaresHomography(from: Data(bytes: source, count: source.count * 8),
                                                       to: Data(bytes: target, count: target.count * 8))
    else { return nil }
    return Homography(matrix.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Array(raw.bindMemory(to: Double.self)) })
}
