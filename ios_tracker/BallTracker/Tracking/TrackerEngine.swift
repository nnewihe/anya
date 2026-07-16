import AVFoundation
import CoreML
import CoreVideo
import Foundation

/// One frame's combined result: raw detections plus the tracker's verdict.
/// Detections and derived positions are in source-frame pixel space.
struct FrameResult {
    let t: Double
    let frameSize: CGSize
    let detections: [BallDetection]
    let status: TrackStatus       // in analysis space (960-wide)
    let analysisScale: CGFloat    // source px * scale = analysis px
    let inferenceMs: Double

    var ballPosition: CGPoint? {
        guard let p = status.position else { return nil }
        return CGPoint(x: p.x / analysisScale, y: p.y / analysisScale)
    }

    /// Smoothed trajectory over the tracker's motion window, in source pixels.
    var trace: [CGPoint] {
        status.trace.map { CGPoint(x: $0.x / analysisScale, y: $0.y / analysisScale) }
    }

    /// Ball speed in source pixels/second.
    var speedPxS: Double { status.speedPxS / Double(analysisScale) }
}

/// Detector + Kalman tracker glued together. The tracker's pixel-tuned gates
/// (gateBasePx etc.) were validated in 960-wide analysis space, so detections
/// are normalized into that space before tracking regardless of the source
/// frame size.
final class TrackerEngine {
    static let analysisWidth: CGFloat = 960

    private let detector: BallDetector
    private var manager: BallTrackManager?
    private var scale: CGFloat = 1
    private let fps: Double
    let confThreshold: Float

    /// Stationary ball-like clutter to drop before tracking, in analysis space.
    /// Empty unless `prepareExclusionZones` has run.
    private(set) var exclusionZones: ExclusionZones = .none

    init(fps: Double,
         conf: Float = BallDetector.defaultConf,
         computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits,
         modelURL: URL? = nil) throws {
        self.detector = try modelURL.map {
            try BallDetector(modelURL: $0, computeUnits: computeUnits)
        } ?? BallDetector(computeUnits: computeUnits)
        self.fps = fps
        self.confThreshold = conf
    }

    func reset() {
        manager = nil
    }

    /// Scan `asset` for stationary ball-like clutter (baskets, balls at rest)
    /// and fence it off, the way anya_base.py does at startup. Reuses this
    /// engine's detector, so it costs one extra pass of ~50 frames, not a
    /// second copy of the model.
    func prepareExclusionZones(asset: AVAsset, frameSize: CGSize) async throws {
        guard frameSize.width > 0 else { return }
        exclusionZones = try await ExclusionZoneScanner.scan(
            asset: asset,
            detector: detector,
            analysisScale: Self.analysisWidth / frameSize.width)
        print("[TrackerEngine] \(exclusionZones.rects.count) exclusion zone(s): "
              + exclusionZones.rects.map {
                  String(format: "(%.0f,%.0f)-(%.0f,%.0f)", $0.minX, $0.minY, $0.maxX, $0.maxY)
              }.joined(separator: " "))
    }

    /// Detect and map into analysis space, dropping exclusion-zone clutter —
    /// no online tracking. The offline video path uses this to gather every
    /// frame's candidates before solving the trajectory globally.
    func analysisDetections(_ pixelBuffer: CVPixelBuffer)
        throws -> (dets: [ViterbiDetection], inferenceMs: Double, frameSize: CGSize) {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        scale = Self.analysisWidth / w

        let t0 = CFAbsoluteTimeGetCurrent()
        let detections = try detector.detect(in: pixelBuffer, conf: confThreshold)
        let inferenceMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let dets = detections.compactMap { d -> ViterbiDetection? in
            let x = Double(d.center.x * scale)
            let y = Double(d.center.y * scale)
            if exclusionZones.contains(x: x, y: y) { return nil }
            return ViterbiDetection(x: x, y: y, conf: Double(d.conf))
        }
        return (dets, inferenceMs, CGSize(width: w, height: h))
    }

    /// Analysis px -> source px, valid once a frame has been seen.
    var analysisScale: CGFloat { scale }

    func process(_ pixelBuffer: CVPixelBuffer, at t: Double) throws -> FrameResult {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let t0 = CFAbsoluteTimeGetCurrent()
        let detections = try detector.detect(in: pixelBuffer, conf: confThreshold)
        let inferenceMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        if manager == nil {
            scale = Self.analysisWidth / w
            manager = BallTrackManager(
                fps: fps,
                perspectiveScale: makeImageRowPerspective(frameHeight: Double(h * scale)))
        }
        // Drop stationary clutter before it reaches the tracker — same filter,
        // same space, as anya_base.py's active-ball candidate loop.
        let tracked = detections.compactMap { d -> TrackerDetection? in
            let x = Double(d.center.x * scale)
            let y = Double(d.center.y * scale)
            if exclusionZones.contains(x: x, y: y) { return nil }
            return TrackerDetection(x: x, y: y, conf: Double(d.conf))
        }
        let status = manager!.update(detections: tracked, now: t)

        return FrameResult(
            t: t,
            frameSize: CGSize(width: w, height: h),
            detections: detections,
            status: status,
            analysisScale: scale,
            inferenceMs: inferenceMs)
    }
}
