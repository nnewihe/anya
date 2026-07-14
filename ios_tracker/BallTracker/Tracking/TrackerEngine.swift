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

    init(fps: Double,
         conf: Float = BallDetector.defaultConf,
         computeUnits: MLComputeUnits = .all,
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
        let tracked = detections.map {
            TrackerDetection(x: Double($0.center.x * scale),
                             y: Double($0.center.y * scale),
                             conf: Double($0.conf))
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
