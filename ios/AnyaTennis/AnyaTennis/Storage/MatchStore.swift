// MatchStore.swift
//
// One folder per match under Application Support/Matches/<id>/:
//   manifest.json          MatchManifest
//   calibration.png        the 960x540 frame the corners were tapped on
//   part-001.mov ...       master recording, one file per part
//   proxies/near-00001.mp4, far-00001.mp4 ...   2 s proxy segments
//   perception.log         PerceptionLog (what analysis has done so far)
//   reel.mov               the highlight reel
// Application Support, not Documents: the files are the app's working state,
// and the user exports what they want through Photos or Share.
import Foundation

enum MatchStore {
    static var root: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let url = base.appendingPathComponent("Matches", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func folder(_ id: String) -> URL { root.appendingPathComponent(id, isDirectory: true) }

    static func file(_ name: String, of id: String) -> URL { folder(id).appendingPathComponent(name) }

    static func create(_ manifest: MatchManifest) throws {
        let proxies = folder(manifest.id).appendingPathComponent("proxies", isDirectory: true)
        try FileManager.default.createDirectory(at: proxies, withIntermediateDirectories: true)
        try save(manifest)
    }

    /// Atomic: a kill mid-write leaves the previous manifest intact.
    static func save(_ manifest: MatchManifest) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: file("manifest.json", of: manifest.id), options: .atomic)
    }

    static func load(_ id: String) throws -> MatchManifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(MatchManifest.self, from: Data(contentsOf: file("manifest.json", of: id)))
    }

    /// Every match, newest first. Folders without a readable manifest are skipped.
    static func all() -> [MatchManifest] {
        let ids = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        return ids.compactMap { try? load($0) }.sorted { $0.createdAt > $1.createdAt }
    }

    static func delete(_ id: String) throws {
        try FileManager.default.removeItem(at: folder(id))
    }

    /// Bytes free on the volume holding the matches.
    static func freeBytes() -> Int64 {
        let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }
}
