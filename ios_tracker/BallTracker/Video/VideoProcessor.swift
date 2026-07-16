import AVFoundation
import Foundation

/// Result of a full offline pass over a video: one sample per decoded frame.
struct VideoAnalysis {
    struct Sample {
        let t: Double
        let pos: CGPoint?    // tracked ball, source px (nil when no track)
        let state: TrackState
        let speedPxS: Double
    }

    let url: URL
    let size: CGSize     // display-oriented
    let fps: Double
    let duration: Double
    let samples: [Sample]
    let avgInferenceMs: Double

    /// Fraction of frames with a live moving/coasting trace.
    var liveTraceRate: Double {
        guard !samples.isEmpty else { return 0 }
        let live = samples.filter { $0.state == .moving || $0.state == .coasting }.count
        return Double(live) / Double(samples.count)
    }

    var maxSpeedPxS: Double { samples.map(\.speedPxS).max() ?? 0 }

    /// Samples in `(t - window, t]`, for drawing the trail at playback time t.
    func trail(at t: Double, window: Double = 0.5) -> [CGPoint] {
        samples
            .filter { $0.t > t - window && $0.t <= t }
            .compactMap { s in
                (s.state == .moving || s.state == .coasting) ? s.pos : nil
            }
    }

    func sample(at t: Double) -> Sample? {
        // Samples are time-ordered; binary search for the last one <= t.
        var lo = 0, hi = samples.count - 1, best = -1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if samples[mid].t <= t { best = mid; lo = mid + 1 } else { hi = mid - 1 }
        }
        return best >= 0 ? samples[best] : nil
    }
}

enum VideoProcessorError: Error {
    case noVideoTrack
    case readFailed
}

/// Decodes a video with AVAssetReader and runs every frame through the same
/// detector + tracker core as the live camera path.
final class VideoProcessor {
    private let modelURL: URL?
    private let viterbiConfig: ViterbiConfig

    /// `modelURL` overrides the app-bundle model (used by the test harness);
    /// `viterbiConfig` lets the tuning sweep vary the solver's weights.
    init(modelURL: URL? = nil, viterbiConfig: ViterbiConfig = ViterbiConfig()) {
        self.modelURL = modelURL
        self.viterbiConfig = viterbiConfig
    }

    func process(url: URL,
                 conf: Float = BallDetector.solverConf,
                 progress: @escaping @Sendable (Double) -> Void) async throws -> VideoAnalysis {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoProcessorError.noVideoTrack
        }
        let duration = try await asset.load(.duration).seconds
        let fps = Double(try await track.load(.nominalFrameRate))

        // A video composition applies the track's preferredTransform, so
        // buffers come out display-oriented and overlay math matches AVPlayer.
        let composition = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
        let size = composition.renderSize

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderVideoCompositionOutput(
            videoTracks: [track],
            videoSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.videoComposition = composition
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw VideoProcessorError.readFailed }
        reader.add(output)
        guard reader.startReading() else { throw VideoProcessorError.readFailed }

        let engine = try TrackerEngine(fps: fps > 0 ? fps : 30, conf: conf, modelURL: modelURL)

        // Fence off stationary ball-like clutter first (ball baskets, balls at
        // rest). Without this a basket — detected on every frame — holds the
        // confirmed track forever and the real ball never gets promoted.
        try await engine.prepareExclusionZones(asset: asset, frameSize: size)

        // Pass 1 — detect every frame. Unlike the live path this commits to no
        // associations: the whole clip's candidates are gathered first so the
        // solver can weigh a link against what happens after it.
        var perFrame: [[ViterbiDetection]] = []
        var times: [Double] = []
        var inferenceTotal = 0.0
        var lastProgressT = -1.0

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

            let (dets, ms, _) = try engine.analysisDetections(pixelBuffer)
            inferenceTotal += ms
            perFrame.append(dets)
            times.append(t)

            if duration > 0, t - lastProgressT > 0.25 {
                lastProgressT = t
                progress(min(t / duration, 1.0) * 0.95)   // leave headroom for the solve
            }
        }

        if reader.status == .failed {
            throw reader.error ?? VideoProcessorError.readFailed
        }

        // Pass 2 — solve the trajectory over the whole clip.
        try Task.checkCancellation()
        let solved = ViterbiBallTracker(config: viterbiConfig).solve(frames: perFrame, times: times)

        // Analysis space -> source px for display.
        let toSource = engine.analysisScale > 0 ? 1 / engine.analysisScale : 1
        let samples = solved.map { s in
            VideoAnalysis.Sample(
                t: s.t,
                pos: s.pos.map { CGPoint(x: $0.x * toSource, y: $0.y * toSource) },
                state: s.state,
                speedPxS: s.speedPxS * Double(toSource))
        }
        progress(1.0)

        return VideoAnalysis(
            url: url,
            size: size,
            fps: fps,
            duration: duration,
            samples: samples,
            avgInferenceMs: times.isEmpty ? 0 : inferenceTotal / Double(times.count))
    }
}
