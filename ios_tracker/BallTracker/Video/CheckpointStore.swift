import CryptoKit
import Foundation

/// On-disk record of a partial (or finished) Pass-1 detection sweep, so an
/// interrupted offline analysis can be resumed after the app is killed.
///
/// Only Pass 1 (per-frame ANE ball detection + Vision player boxes) is
/// checkpointed — it's the expensive part. Pass 2 (the streaming tracker
/// replay + rally segmentation) is cheap and re-runs from the restored
/// detections. Detections depend only on the detector inputs (models, `conf`,
/// analysis width, exclusion zones), never on the rally post-processing, so a
/// checkpoint stays valid across segmentation tuning; `detectorFingerprint`
/// guards the rest.
struct ProcessingCheckpoint: Codable {
    static let currentSchema = 2

    var schemaVersion = ProcessingCheckpoint.currentSchema
    let videoKey: String
    let workingCopyName: String
    let displayName: String
    let detectorFingerprint: String

    let fps: Double
    let width: Double
    let height: Double
    let duration: Double

    /// Exclusion zones in analysis space, each `[minX, minY, maxX, maxY]`.
    let zones: [[Double]]

    /// Presentation time of the last frame whose detections are recorded.
    var lastFrameTime: Double
    var frameCount: Int
    var inferenceTotalMs: Double

    /// Parallel arrays, one entry per processed frame.
    var times: [Double]
    var frames: [[TrackerDetection]]
    var players: [PlayerBoxes]

    var completed: Bool

    /// Fraction of the clip whose detections are already recorded.
    var progress: Double { duration > 0 ? min(lastFrameTime / duration, 1) : 0 }
}

/// What the resume prompt needs to offer to continue an interrupted analysis.
struct ResumeInfo: Identifiable {
    let key: String
    let workingCopyURL: URL
    let displayName: String
    let progress: Double
    var id: String { key }
}

/// Manages the persistent working copies and checkpoint files under
/// Application Support, and the atomic reads/writes between them.
///
/// Stateless apart from its (immutable) directory URLs, so it is safe to hand
/// to the detached processing task.
final class CheckpointStore: Sendable {
    private let videosDir: URL
    private let checkpointsDir: URL

    init() {
        let base = (try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        let root = base.appendingPathComponent("BallTracker", isDirectory: true)
        videosDir = root.appendingPathComponent("videos", isDirectory: true)
        checkpointsDir = root.appendingPathComponent("checkpoints", isDirectory: true)
        for dir in [videosDir, checkpointsDir] {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    // MARK: Identity + working copy

    /// Stable key for a video: SHA-256 over its byte length plus the first and
    /// last 64 KB. Cheap (no full read) and stable across app launches, so the
    /// same picked asset maps to the same checkpoint every time.
    func fingerprint(of fileURL: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        let size = (try FileManager.default.attributesOfItem(atPath: fileURL.path)[.size]
            as? Int) ?? 0

        var hasher = SHA256()
        withUnsafeBytes(of: Int64(size).littleEndian) { hasher.update(bufferPointer: $0) }

        let chunk = 64 * 1024
        if let head = try handle.read(upToCount: chunk) { hasher.update(data: head) }
        if size > chunk {
            try handle.seek(toOffset: UInt64(max(0, size - chunk)))
            if let tail = try handle.read(upToCount: chunk) { hasher.update(data: tail) }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Move an imported temp file to a persistent, key-addressed location — or,
    /// if a copy for this key already exists (same asset picked again after a
    /// relaunch), drop the temp file and reuse the existing one.
    func adoptWorkingCopy(tempURL: URL, key: String, ext: String) throws -> URL {
        let dest = videosDir.appendingPathComponent(key)
            .appendingPathExtension(ext.isEmpty ? "mov" : ext)
        let fm = FileManager.default
        if fm.fileExists(atPath: dest.path) {
            try? fm.removeItem(at: tempURL)
        } else {
            try fm.moveItem(at: tempURL, to: dest)
        }
        return dest
    }

    // MARK: Checkpoint read/write

    private func checkpointURL(key: String) -> URL {
        checkpointsDir.appendingPathComponent(key).appendingPathExtension("json")
    }

    func load(key: String) -> ProcessingCheckpoint? {
        guard let data = try? Data(contentsOf: checkpointURL(key: key)),
              let cp = try? JSONDecoder().decode(ProcessingCheckpoint.self, from: data),
              cp.schemaVersion == ProcessingCheckpoint.currentSchema
        else { return nil }
        return cp
    }

    /// Atomic write: encode to a sibling `.tmp` then replace, so a crash mid-write
    /// can never leave a half-written checkpoint.
    func save(_ checkpoint: ProcessingCheckpoint) throws {
        let dest = checkpointURL(key: checkpoint.videoKey)
        let tmp = dest.appendingPathExtension("tmp")
        let data = try JSONEncoder().encode(checkpoint)
        try data.write(to: tmp, options: .atomic)
        let fm = FileManager.default
        if fm.fileExists(atPath: dest.path) {
            _ = try fm.replaceItemAt(dest, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: dest)
        }
    }

    func markCompleted(key: String) {
        guard var cp = load(key: key) else { return }
        cp.completed = true
        try? save(cp)
    }

    /// Remove a checkpoint and its working copy (user chose "Discard", or the
    /// analysis finished and playback no longer needs the record).
    func discard(key: String) {
        let fm = FileManager.default
        if let cp = load(key: key) {
            try? fm.removeItem(at: videosDir.appendingPathComponent(cp.workingCopyName))
        }
        try? fm.removeItem(at: checkpointURL(key: key))
    }

    private func workingCopyURL(name: String) -> URL {
        videosDir.appendingPathComponent(name)
    }

    // MARK: Resume + pruning

    /// The most recent unfinished analysis whose working copy still exists and
    /// whose detector inputs still match — the one to offer resuming.
    func resumable(detectorFingerprint: String) -> ResumeInfo? {
        allCheckpoints()
            .filter { !$0.cp.completed
                && $0.cp.detectorFingerprint == detectorFingerprint
                && FileManager.default.fileExists(
                    atPath: workingCopyURL(name: $0.cp.workingCopyName).path) }
            .max { $0.modified < $1.modified }
            .map { entry in
                ResumeInfo(
                    key: entry.cp.videoKey,
                    workingCopyURL: workingCopyURL(name: entry.cp.workingCopyName),
                    displayName: entry.cp.displayName,
                    progress: entry.cp.progress)
            }
    }

    /// Bound storage: keep every unfinished checkpoint (and its working copy)
    /// plus the single most recent completed one; drop older completed records
    /// and any working copy no checkpoint references.
    func prune() {
        let all = allCheckpoints()
        let completed = all.filter { $0.cp.completed }.sorted { $0.modified > $1.modified }
        for stale in completed.dropFirst() { discard(key: stale.cp.videoKey) }

        // Sweep orphaned working copies (no checkpoint points at them).
        let referenced = Set(allCheckpoints().map(\.cp.workingCopyName))
        let fm = FileManager.default
        let files = (try? fm.contentsOfDirectory(at: videosDir,
            includingPropertiesForKeys: nil)) ?? []
        for file in files where !referenced.contains(file.lastPathComponent) {
            try? fm.removeItem(at: file)
        }
    }

    private func allCheckpoints() -> [(cp: ProcessingCheckpoint, modified: Date)] {
        let fm = FileManager.default
        let files = (try? fm.contentsOfDirectory(at: checkpointsDir,
            includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return files.compactMap { url in
            guard url.pathExtension == "json",
                  let data = try? Data(contentsOf: url),
                  let cp = try? JSONDecoder().decode(ProcessingCheckpoint.self, from: data)
            else { return nil }
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            return (cp, modified)
        }
    }
}
