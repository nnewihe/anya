// FileReplay.swift  (Debug builds only)
//
// The simulator has no camera. This plays a video file into CameraSession's
// frame callback exactly as the camera would -- same queue, same pixel
// format, the file's own timestamps -- so the whole recording path
// (master, proxies, live analysis, reel) runs end to end on a known clip.
//
//   -AnyaReplay /path/clip.mp4 -AnyaCorners "x1,y1,x2,y2,x3,y3,x4,y4" [-AnyaReplaySpeed 4]
//
// Corners are the four singles corners in 960x540 analysis px, in the
// calibration order (near-left, near-right, far-right, far-left): a corpus
// clip's <stem>_court_cache.json "points".
#if DEBUG
import AVFoundation

struct ReplayRequest {
    let url: URL
    let corners: [CGPoint]
    let speed: Double

    static func fromLaunchArguments() -> ReplayRequest? {
        let defaults = UserDefaults.standard
        guard let path = defaults.string(forKey: "AnyaReplay"),
              let numbers = defaults.string(forKey: "AnyaCorners")?.split(separator: ",").compactMap({ Double($0) }),
              numbers.count == 8 else { return nil }
        let corners = stride(from: 0, to: 8, by: 2).map { CGPoint(x: numbers[$0], y: numbers[$0 + 1]) }
        let speed = defaults.double(forKey: "AnyaReplaySpeed")
        return ReplayRequest(url: URL(fileURLWithPath: path), corners: corners, speed: speed > 0 ? speed : 1)
    }
}

extension CameraSession: @unchecked Sendable {}   // used only on its own queue

enum FileReplay {
    /// The file's size and frame rate.
    static func format(of url: URL) async throws -> (width: Int, height: Int, fps: Double) {
        let track = try await AVURLAsset(url: url).loadTracks(withMediaType: .video).first!
        let size = try await track.load(.naturalSize)
        return (Int(size.width), Int(size.height), Double(try await track.load(.nominalFrameRate)))
    }

    /// The calibration frame: the one at 10 s, as camera.reference_index
    /// picks frame 300 (and some files open on a black dummy frame).
    static func calibrationFrame(of url: URL, at seconds: Double = 10) async throws -> CVPixelBuffer {
        // Copied: the buffer must outlive the reader, which is freed on return.
        let reader = try await makeReader(url, copying: true)
        var last: CVPixelBuffer?
        while let sample = reader.output.copyNextSampleBuffer() {
            last = CMSampleBufferGetImageBuffer(sample) ?? last
            if CMSampleBufferGetPresentationTimeStamp(sample).seconds >= seconds { break }
        }
        guard let last else { throw CocoaError(.fileReadCorruptFile) }
        return last
    }

    /// Push every frame into `session.onVideo` on the capture queue, paced at
    /// `speed` x real time. Returns when the file ends.
    static func play(_ url: URL, into session: CameraSession, speed: Double) async throws {
        let reader = try await makeReader(url)
        var first: CMTime?
        let started = Date()
        while let sample = reader.output.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            if first == nil { first = pts }
            let due = (pts - first!).seconds / speed - Date().timeIntervalSince(started)
            if due > 0 { try await Task.sleep(for: .seconds(due)) }
            nonisolated(unsafe) let frame = sample
            await withCheckedContinuation { continuation in
                session.queue.async {
                    session.onVideo?(frame)
                    continuation.resume()
                }
            }
        }
    }

    private static func makeReader(_ url: URL, copying: Bool = false) async throws -> (reader: AVAssetReader, output: AVAssetReaderTrackOutput) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw CocoaError(.fileReadCorruptFile) }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange])
        output.alwaysCopiesSampleData = copying
        reader.add(output)
        reader.startReading()
        return (reader, output)
    }
}
#endif
