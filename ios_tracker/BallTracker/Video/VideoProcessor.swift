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

    /// `modelURL` overrides the app-bundle model (used by the test harness).
    init(modelURL: URL? = nil) {
        self.modelURL = modelURL
    }

    func process(url: URL,
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

        let engine = try TrackerEngine(fps: fps > 0 ? fps : 30, modelURL: modelURL)

        var samples: [VideoAnalysis.Sample] = []
        var inferenceTotal = 0.0
        var lastProgressT = -1.0

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

            let result = try engine.process(pixelBuffer, at: t)
            inferenceTotal += result.inferenceMs

            let live = result.status.state == .moving || result.status.state == .coasting
            samples.append(VideoAnalysis.Sample(
                t: t,
                pos: live ? result.ballPosition : nil,
                state: result.status.state,
                speedPxS: result.speedPxS))

            if duration > 0, t - lastProgressT > 0.25 {
                lastProgressT = t
                progress(min(t / duration, 1.0))
            }
        }

        if reader.status == .failed {
            throw reader.error ?? VideoProcessorError.readFailed
        }
        progress(1.0)

        return VideoAnalysis(
            url: url,
            size: size,
            fps: fps,
            duration: duration,
            samples: samples,
            avgInferenceMs: samples.isEmpty ? 0 : inferenceTotal / Double(samples.count))
    }
}
