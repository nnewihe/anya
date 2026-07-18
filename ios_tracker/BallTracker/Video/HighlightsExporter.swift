import AVFoundation
import Foundation

/// A contiguous keep-range of the source video, in seconds.
struct HighlightSegment: Identifiable {
    let start: Double
    let end: Double
    let origin: RallyOrigin
    var id: Double { start }
    var duration: Double { end - start }
}

/// Stitches the rally detector's segments into a highlights reel.
///
/// The keep-ranges are exactly the segments `RallyAccumulator` cut during the
/// offline analysis (pipeline/rally_detector.py's rules: trace-driven starts,
/// end-pad, gap merge, HMM serving-pattern filter, pre-roll) — this type only
/// does the cutting. The stitch itself uses AVFoundation (AVMutableComposition
/// + export) rather than ffmpeg, since the app can't shell out — but the kept
/// time ranges match pipeline/utilities.py's create_highlights_ffmpeg output.
enum HighlightsExporter {
    enum ExportError: LocalizedError {
        case noSegments
        case noVideoTrack
        case exportFailed(String)

        var errorDescription: String? {
            switch self {
            case .noSegments:
                return "No rally segments detected in this video."
            case .noVideoTrack:
                return "The source video has no video track."
            case .exportFailed(let m):
                return "Export failed: \(m)"
            }
        }
    }

    /// The analysis' rally segments in UI form. Pure and cheap — safe to call
    /// on the main actor to preview counts before committing to an export.
    static func segments(for analysis: VideoAnalysis) -> [HighlightSegment] {
        analysis.rallySegments.map {
            HighlightSegment(start: $0.start, end: $0.end, origin: $0.origin)
        }
    }

    /// Build the reel and return the output file URL. Reuses the source asset's
    /// video (and audio, if present) so no re-encode of the picture is needed.
    static func export(analysis: VideoAnalysis) async throws -> URL {
        let segs = segments(for: analysis)
        guard !segs.isEmpty else { throw ExportError.noSegments }

        let asset = AVURLAsset(url: analysis.url)
        guard let srcVideo = try await asset.loadTracks(withMediaType: .video).first else {
            throw ExportError.noVideoTrack
        }
        let srcAudio = try await asset.loadTracks(withMediaType: .audio).first

        let comp = AVMutableComposition()
        guard let compVideo = comp.addMutableTrack(
            withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ExportError.exportFailed("could not create composition track")
        }
        // Carry the source orientation so portrait clips aren't sideways.
        compVideo.preferredTransform = try await srcVideo.load(.preferredTransform)
        let compAudio = srcAudio == nil ? nil : comp.addMutableTrack(
            withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)

        let scale = try await asset.load(.duration).timescale
        var cursor = CMTime.zero
        for seg in segs {
            let range = CMTimeRange(
                start: CMTime(seconds: seg.start, preferredTimescale: scale),
                end: CMTime(seconds: seg.end, preferredTimescale: scale))
            try compVideo.insertTimeRange(range, of: srcVideo, at: cursor)
            if let compAudio, let srcAudio {
                try? compAudio.insertTimeRange(range, of: srcAudio, at: cursor)
            }
            cursor = cursor + range.duration
        }

        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent(analysis.url.deletingPathExtension().lastPathComponent
                                    + "_highlights_\(Int(Date().timeIntervalSince1970)).mp4")
        guard let session = AVAssetExportSession(
            asset: comp, presetName: AVAssetExportPresetHighestQuality) else {
            throw ExportError.exportFailed("could not create export session")
        }
        session.outputURL = out
        session.outputFileType = .mp4
        session.shouldOptimizeForNetworkUse = true

        await session.export()
        switch session.status {
        case .completed:
            return out
        case .cancelled:
            throw ExportError.exportFailed("cancelled")
        default:
            throw ExportError.exportFailed(session.error?.localizedDescription ?? "unknown")
        }
    }
}
