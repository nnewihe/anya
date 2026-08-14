import AVFoundation
import CoreML
import CoreVideo
import Foundation

/// Detector + Kalman tracker glued together. The tracker's pixel-tuned gates
/// (gateBasePx etc.) were validated in 1920-wide analysis space, so detections
/// are normalized into that space before tracking regardless of the source
/// frame size.
final class TrackerEngine {
    static let analysisWidth: CGFloat = 1920

    /// Lazily constructed: with a ROI model active, Live mode never touches
    /// this (no exclusion-zone scan runs there), and loading it eagerly meant
    /// TrackerEngine.init synchronously compiled TWO Core ML models on the
    /// main actor before the first camera frame — slow enough on the
    /// Simulator's CPU-only path to blow the launch watchdog and get the app
    /// killed with no crash report. Deferred to first actual use instead.
    private let fullModelURL: URL?
    private let fullComputeUnits: MLComputeUnits
    private var _detector: BallDetector?
    private func fullDetector() throws -> BallDetector {
        if let d = _detector { return d }
        let d = try fullModelURL.map {
            try BallDetector(modelURL: $0, computeUnits: fullComputeUnits)
        } ?? BallDetector(computeUnits: fullComputeUnits)
        _detector = d
        return d
    }

    private var scale: CGFloat = 1
    private let fps: Double
    let confThreshold: Float

    /// Small-input detector + planner for the tracked-ROI path. When present,
    /// per-frame detection runs on planned crops instead of the full frame
    /// (`detector` is then only used for the exclusion-zone scan).
    private let roiDetector: BallDetector?
    private(set) var roiPlanner: RoiPlanner?
    /// Steering tracker for the offline pass-1 path, which gathers detections
    /// without tracking (Pass 2 replays them): the ROI needs *some* track to
    /// follow during Pass 1, so a private manager shadows what Pass 2 will do.
    private var steer: BallTrackManager?
    private var steerStatus: TrackStatus?

    /// Stationary ball-like clutter to drop before tracking, in analysis space.
    /// Empty unless `prepareExclusionZones` has run.
    private(set) var exclusionZones: ExclusionZones = .none

    init(fps: Double,
         conf: Float = BallDetector.defaultConf,
         computeUnits: MLComputeUnits = BallDetector.defaultComputeUnits,
         modelURL: URL? = nil,
         roiModelURL: URL? = nil) throws {
        self.fullModelURL = modelURL
        self.fullComputeUnits = computeUnits
        if let roiModelURL {
            let roi = try BallDetector(modelURL: roiModelURL, computeUnits: computeUnits)
            self.roiDetector = roi
            self.roiPlanner = RoiPlanner(inputWidth: roi.inputWidth,
                                         inputHeight: roi.inputHeight)
        } else {
            self.roiDetector = nil
        }
        self.fps = fps
        self.confThreshold = conf
    }

    /// Detect via the planned crops (tracked ROI or tiered scan), returning
    /// deduped detections in source-frame pixels.
    private func roiDetect(_ pixelBuffer: CVPixelBuffer, detector roi: BallDetector,
                           planner: RoiPlanner, frameSize: CGSize,
                           steering: TrackStatus?) throws -> [BallDetection] {
        let crops = planner.crops(frameSize: frameSize,
                                  analysisScale: scale,
                                  fps: fps,
                                  status: steering)
        var all: [BallDetection] = []
        for c in crops {
            all += try roi.detect(in: pixelBuffer, conf: confThreshold,
                                  maxBoxPx: c.maxBoxPx, crop: c.rect)
        }
        return RoiPlanner.dedup(all)
    }

    /// Scan `asset` for stationary ball-like clutter (baskets, balls at rest)
    /// and fence it off, the way anya_base.py does at startup. Reuses this
    /// engine's detector, so it costs one extra pass of ~50 frames, not a
    /// second copy of the model.
    func prepareExclusionZones(asset: AVAsset, frameSize: CGSize) async throws {
        guard frameSize.width > 0 else { return }
        exclusionZones = try await ExclusionZoneScanner.scan(
            asset: asset,
            detector: try fullDetector(),
            analysisScale: Self.analysisWidth / frameSize.width)
        print("[TrackerEngine] \(exclusionZones.rects.count) exclusion zone(s): "
              + exclusionZones.rects.map {
                  String(format: "(%.0f,%.0f)-(%.0f,%.0f)", $0.minX, $0.minY, $0.maxX, $0.maxY)
              }.joined(separator: " "))
    }

    /// Restore exclusion zones computed on a previous run (from a checkpoint),
    /// so a resumed analysis skips the scan pass and fences off exactly the same
    /// clutter it did the first time.
    func setExclusionZones(_ zones: ExclusionZones) {
        exclusionZones = zones
    }

    /// Detect and map into analysis space, dropping exclusion-zone clutter —
    /// no *authoritative* tracking. The offline video path uses this to gather
    /// every frame's candidates before replaying them through the streaming
    /// tracker. `t` drives the internal steering tracker the ROI path follows;
    /// full-frame mode ignores it.
    func analysisDetections(_ pixelBuffer: CVPixelBuffer, at t: Double = 0)
        throws -> (dets: [TrackerDetection], inferenceMs: Double, frameSize: CGSize) {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        scale = Self.analysisWidth / w

        let t0 = CFAbsoluteTimeGetCurrent()
        let detections: [BallDetection]
        if let roi = roiDetector, let planner = roiPlanner {
            detections = try roiDetect(pixelBuffer, detector: roi, planner: planner,
                                       frameSize: CGSize(width: w, height: h),
                                       steering: steerStatus)
        } else {
            detections = try fullDetector().detect(in: pixelBuffer, conf: confThreshold)
        }
        let inferenceMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let dets = detections.compactMap { d -> TrackerDetection? in
            let x = Double(d.center.x * scale)
            let y = Double(d.center.y * scale)
            if exclusionZones.contains(x: x, y: y) { return nil }
            return TrackerDetection(x: x, y: y, conf: Double(d.conf))
        }
        if roiDetector != nil {
            if steer == nil {
                steer = BallTrackManager(
                    fps: fps,
                    perspectiveScale: makeImageRowPerspective(frameHeight: Double(h * scale)))
            }
            steerStatus = steer!.update(detections: dets, now: t)
        }
        return (dets, inferenceMs, CGSize(width: w, height: h))
    }
}
