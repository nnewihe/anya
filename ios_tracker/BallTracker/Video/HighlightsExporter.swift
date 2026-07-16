import AVFoundation
import Foundation

/// A contiguous keep-range of the source video, in seconds.
struct HighlightSegment: Identifiable {
    let start: Double
    let end: Double
    var id: Double { start }
    var duration: Double { end - start }
}

/// Turns a `VideoAnalysis` trace into a stitched highlights reel.
///
/// The segment logic is the exact port of ios_tracker/make_highlights.py:
///   * a "live span" is a contiguous run of moving/coasting frames — the same
///     definition `liveTraceRate` reports,
///   * keep spans at least `minLen` seconds long (verified trace),
///   * merge kept spans whose gap is <= `mergeGap` seconds into one cut,
///   * pad each cut by `pad` seconds for watchability.
///
/// The stitch itself uses AVFoundation (AVMutableComposition + export) rather
/// than ffmpeg, since the app can't shell out — but the kept time ranges are
/// identical to the command-line tool's.
enum HighlightsExporter {
    struct Params {
        var minLen: Double = 1.5
        var mergeGap: Double = 3.0
        var pad: Double = 0.5
    }

    enum ExportError: LocalizedError {
        case noSegments
        case noVideoTrack
        case exportFailed(String)

        var errorDescription: String? {
            switch self {
            case .noSegments:
                return "No rallies long enough to include (need a tracked span ≥ 1.5s)."
            case .noVideoTrack:
                return "The source video has no video track."
            case .exportFailed(let m):
                return "Export failed: \(m)"
            }
        }
    }

    /// Compute keep-segments from the analysis. Pure and cheap — safe to call on
    /// the main actor to preview counts before committing to an export.
    static func segments(for analysis: VideoAnalysis,
                         params: Params = Params()) -> [HighlightSegment] {
        // 1. Contiguous live spans.
        var spans: [(Double, Double)] = []
        var runStart: Double?
        var prevT = 0.0
        for s in analysis.samples {
            let live = s.state == .moving || s.state == .coasting
            if live, runStart == nil {
                runStart = s.t
            } else if !live, let rs = runStart {
                spans.append((rs, prevT))
                runStart = nil
            }
            prevT = s.t
        }
        if let rs = runStart { spans.append((rs, prevT)) }

        // 2. Keep the long-enough spans.
        let kept = spans.filter { $0.1 - $0.0 >= params.minLen }.sorted { $0.0 < $1.0 }
        guard !kept.isEmpty else { return [] }

        // 3. Merge within mergeGap.
        var merged: [[Double]] = [[kept[0].0, kept[0].1]]
        for (s, e) in kept.dropFirst() {
            if s - merged[merged.count - 1][1] <= params.mergeGap {
                merged[merged.count - 1][1] = max(merged[merged.count - 1][1], e)
            } else {
                merged.append([s, e])
            }
        }

        // 4. Pad, clamp to the clip, and re-merge any overlap the pad created.
        var out: [HighlightSegment] = []
        for m in merged {
            let s = max(0, m[0] - params.pad)
            let e = min(analysis.duration, m[1] + params.pad)
            if let last = out.last, s <= last.end {
                out[out.count - 1] = HighlightSegment(start: last.start, end: max(last.end, e))
            } else {
                out.append(HighlightSegment(start: s, end: e))
            }
        }
        return out
    }

    /// Build the reel and return the output file URL. Reuses the source asset's
    /// video (and audio, if present) so no re-encode of the picture is needed.
    static func export(analysis: VideoAnalysis,
                       params: Params = Params()) async throws -> URL {
        let segs = segments(for: analysis, params: params)
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
