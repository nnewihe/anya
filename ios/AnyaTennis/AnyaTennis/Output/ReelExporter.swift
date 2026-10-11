// ReelExporter.swift
//
// Cuts the highlight reel out of the master recording: the orchestrator's
// segments, in order, butted together. PASSTHROUGH -- the picture is not
// re-encoded, so a 2-hour match's reel is ready in seconds and keeps full
// quality. (A cut lands on the key frame before it; with a key frame every
// second the .mov's edit list hides the difference.)
//
// The orchestrator's times are on the match TIMELINE: all parts joined end
// to end. A segment can straddle the seam between two parts; it is then cut
// from both.
import Anya2Engine
import AVFoundation

enum ReelExporter {
    enum ExportError: LocalizedError {
        case noPoints, noVideo, failed(String)
        var errorDescription: String? {
            switch self {
            case .noPoints: "No points were found in this recording."
            case .noVideo: "The recording has no video."
            case .failed(let message): "Making the reel failed: \(message)"
            }
        }
    }

    /// Timeline seconds -> (part, seconds into that part's file) pieces.
    static func pieces(of segments: [(start: Double, end: Double)], parts: [MatchManifest.Part],
                       fps: Double) -> [(part: Int, start: Double, end: Double)] {
        var out: [(Int, Double, Double)] = []
        for segment in segments {
            for (i, part) in parts.enumerated() {
                let partStart = Double(part.firstSourceFrame) / fps
                let partEnd = partStart + Double(part.frameCount) / fps
                let start = max(segment.start, partStart), end = min(segment.end, partEnd)
                if end > start { out.append((i, start - partStart, end - partStart)) }
            }
        }
        return out
    }

    static func export(manifest: MatchManifest, segments: [(start: Double, end: Double)], to url: URL) async throws {
        let cuts = pieces(of: segments, parts: manifest.parts, fps: manifest.sourceFPS)
        guard !cuts.isEmpty else { throw ExportError.noPoints }
        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(withMediaType: .video,
                                                           preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ExportError.failed("could not make a video track")
        }
        let audioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
        var sources: [Int: (video: AVAssetTrack, audio: AVAssetTrack?)] = [:]
        var cursor = CMTime.zero
        for cut in cuts {
            if sources[cut.part] == nil {
                let asset = AVURLAsset(url: MatchStore.file(manifest.parts[cut.part].file, of: manifest.id))
                guard let video = try await asset.loadTracks(withMediaType: .video).first else { throw ExportError.noVideo }
                sources[cut.part] = (video, try await asset.loadTracks(withMediaType: .audio).first)
            }
            let source = sources[cut.part]!
            let range = CMTimeRange(start: CMTime(seconds: cut.start, preferredTimescale: 600),
                                    end: CMTime(seconds: cut.end, preferredTimescale: 600))
            try videoTrack.insertTimeRange(range, of: source.video, at: cursor)
            if let audio = source.audio { try? audioTrack?.insertTimeRange(range, of: audio, at: cursor) }
            cursor = cursor + range.duration
        }
        if audioTrack?.segments.isEmpty ?? false, let audioTrack { composition.removeTrack(audioTrack) }

        try? FileManager.default.removeItem(at: url)
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw ExportError.failed("could not start the export")
        }
        session.outputURL = url
        session.outputFileType = .mov
        await session.export()
        if session.status != .completed {
            throw ExportError.failed(session.error?.localizedDescription ?? "status \(session.status.rawValue)")
        }
    }
}
