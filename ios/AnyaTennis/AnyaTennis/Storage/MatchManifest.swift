// MatchManifest.swift
//
// Everything about one match that must survive the app being killed:
// calibration, the recording's parts, how far analysis got, and the outputs.
// Saved as manifest.json in the match's folder (MatchStore), atomically, on
// every state change.
import Foundation

struct MatchManifest: Codable, Identifiable {
    enum State: String, Codable {
        case calibrated     // corners tapped, nothing recorded yet
        case recording      // a part is being written (or the app died mid-part)
        case paused         // between parts: interrupted, can resume recording
        case finishing      // Stop tapped: analysis draining, reel being cut
        case finished       // reel ready
        case failed         // finishing failed; can be retried
    }

    struct Point: Codable { var x: Double; var y: Double }

    /// The far band crop, in source pixels (FarBand).
    struct Band: Codable { var x1: Int; var y1: Int; var x2: Int; var y2: Int }

    /// One continuous stretch of recording. Parts are joined end to end into
    /// one timeline; `firstSourceFrame` is where this part starts on it.
    struct Part: Codable {
        var file: String                 // master movie, in the match folder
        var firstSourceFrame: Int
        var frameCount: Int              // frames written (0 while recording)
        var startedAt: Date
    }

    /// One proxy segment: `count` pose samples starting at `firstSourceFrame`.
    struct Segment: Codable {
        var index: Int
        var firstSourceFrame: Int
        var count: Int
        var nearFile: String
        var farFile: String
    }

    struct Info: Codable {
        var title = ""
        var players = ["", "", "", ""]   // near pair, far pair (singles: first of each)
        var location = ""
        var notes = ""
    }

    struct ReelSummary: Codable {
        var file: String
        var points: Int
        var seconds: Double
        var nearServes: Int
        var farServes: Int
        var pointEnds: Int
    }

    struct Upload: Codable {
        var sessionURL: URL?             // YouTube resumable session
        var videoID: String?
        var privacy = "private"
    }

    var id: String
    var createdAt: Date
    var state: State
    var sourceWidth: Int
    var sourceHeight: Int
    var sourceFPS: Double
    var courtCorners: [Point]            // analysis px (960x540): near-left, near-right, far-right, far-left
    var band: Band
    var parts: [Part] = []
    var segments: [Segment] = []
    var samplesAnalysed = 0              // pose samples in the perception log
    var info = Info()
    var reel: ReelSummary?
    var savedReelToPhotos = false
    var savedRawToPhotos = false
    var upload = Upload()
    var lastError: String?

    /// The timeline's length so far, in source frames.
    var totalFrames: Int { parts.last.map { $0.firstSourceFrame + $0.frameCount } ?? 0 }

    var displayTitle: String {
        if !info.title.isEmpty { return info.title }
        return createdAt.formatted(date: .abbreviated, time: .shortened)
    }
}

// ── what a manifest read from disk may contain ───────────────────────────
//
// A manifest is trusted input only after this: a match folder can come back
// from an edited backup, and its fields name files, build file paths and
// size buffers. Anything outside what the app itself writes is refused.
extension MatchManifest {
    enum Invalid: Error { case field(String) }

    func validated(folderName: String) throws -> MatchManifest {
        func require(_ ok: Bool, _ field: String) throws { if !ok { throw Invalid.field(field) } }
        try require(UUID(uuidString: id) != nil && id == folderName, "id")
        try require(sourceFPS.isFinite && sourceFPS > 1 && sourceFPS <= 240, "sourceFPS")
        try require((16...8192).contains(sourceWidth) && (16...8192).contains(sourceHeight), "source size")
        try require(courtCorners.count == 4 && courtCorners.allSatisfy {
            $0.x.isFinite && $0.y.isFinite && (0...960).contains($0.x) && (0...540).contains($0.y)
        }, "courtCorners")
        try require(band.x1 >= 0 && band.y1 >= 0 && band.x2 <= sourceWidth && band.y2 <= sourceHeight
                    && band.x2 - band.x1 >= 32 && band.y2 - band.y1 >= 32, "band")
        try require(parts.allSatisfy { Self.matches($0.file, #"^part-\d{3}\.mov$"#)
                                       && $0.firstSourceFrame >= 0 && $0.frameCount >= 0 }, "parts")
        try require(segments.allSatisfy { Self.matches($0.nearFile, #"^proxies/near-\d{5}\.mp4$"#)
                                          && Self.matches($0.farFile, #"^proxies/far-\d{5}\.mp4$"#)
                                          && $0.firstSourceFrame >= 0 && (0...1000).contains($0.count) }, "segments")
        try require(reel == nil || reel!.file == "reel.mov", "reel")
        try require(upload.videoID == nil || Self.matches(upload.videoID!, #"^[A-Za-z0-9_-]{11}$"#), "videoID")
        try require(upload.sessionURL == nil || YouTubeUploader.isSessionURL(upload.sessionURL!), "sessionURL")
        try require(["private", "unlisted", "public"].contains(upload.privacy), "privacy")
        return self
    }

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        text.range(of: pattern, options: .regularExpression) != nil
    }
}
