// PerceptionLog.swift
//
// The durable record of analysis: one entry per pose sample, appended as it
// is analysed. After a kill, replaying the log rebuilds the live analysis in
// seconds instead of re-running pose on everything recorded so far.
//
// Entry layout (little-endian):
//   int32   source frame
//   uint8   camera: 0 = not a camera sample, 1 = registration failed, 2 = warp follows
//   9 x f64 the warp (only when camera == 2)
//   uint8   near people, uint8 far people
//   per person: 4 x f32 box, f32 confidence, 51 x f32 keypoints
// A kill mid-append leaves a torn last entry; reading stops before it, and
// the next append starts by cutting the file back to the last whole entry.
import Anya2Engine
import Foundation

public struct LoggedSample {
    public let sourceFrame: Int
    /// nil = not a camera sample; .some(nil) = registration failed.
    public let cameraWarp: Homography??
    public let perceived: PerceivedSample
}

public final class PerceptionLog {
    public let url: URL
    private var handle: FileHandle?

    public init(url: URL) { self.url = url }

    deinit { try? handle?.close() }

    /// Every whole entry in the file, and the byte length they cover.
    public func readAll() -> (samples: [LoggedSample], validBytes: Int) {
        guard let data = try? Data(contentsOf: url) else { return ([], 0) }
        var samples: [LoggedSample] = []
        var offset = 0
        while let (sample, length) = PerceptionLog.decode(data, at: offset) {
            samples.append(sample)
            offset += length
        }
        return (samples, offset)
    }

    /// Append entries and flush them to disk.
    public func append(_ samples: [LoggedSample]) throws {
        if handle == nil {
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let valid = readAll().validBytes
            let opened = try FileHandle(forWritingTo: url)
            try opened.truncate(atOffset: UInt64(valid))   // drop a torn tail
            try opened.seekToEnd()
            handle = opened
        }
        var data = Data()
        for sample in samples { PerceptionLog.encode(sample, into: &data) }
        try handle!.write(contentsOf: data)
        try handle!.synchronize()
    }

    // ── encoding ──────────────────────────────────────────────────────────

    static func encode(_ sample: LoggedSample, into data: inout Data) {
        func put<T>(_ value: T) { withUnsafeBytes(of: value) { data.append(contentsOf: $0) } }
        put(Int32(sample.sourceFrame).littleEndian)
        switch sample.cameraWarp {
        case .none: put(UInt8(0))
        case .some(.none): put(UInt8(1))
        case .some(.some(let warp)):
            put(UInt8(2))
            for value in warp.m { put(value.bitPattern.littleEndian) }
        }
        for row in [sample.perceived.near, sample.perceived.far] {
            put(UInt8(row.confidences.prefix { !$0.isNaN }.count))
        }
        for row in [sample.perceived.near, sample.perceived.far] {
            let people = row.confidences.prefix { !$0.isNaN }.count
            for person in 0..<people {
                for value in row.boxes[(person * 4)..<(person * 4 + 4)] { put(value.bitPattern.littleEndian) }
                put(row.confidences[person].bitPattern.littleEndian)
                for value in row.keypoints[(person * 51)..<(person * 51 + 51)] { put(value.bitPattern.littleEndian) }
            }
        }
    }

    static func decode(_ data: Data, at start: Int) -> (LoggedSample, Int)? {
        var offset = start
        func get<T: FixedWidthInteger>(_: T.Type) -> T? {
            let size = MemoryLayout<T>.size
            guard offset + size <= data.count else { return nil }
            var value: T = 0
            withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0, from: (data.startIndex + offset)..<(data.startIndex + offset + size)) }
            offset += size
            return T(littleEndian: value)
        }
        func float() -> Float? { get(UInt32.self).map { Float(bitPattern: $0) } }
        guard let frame = get(Int32.self), let cameraKind = get(UInt8.self) else { return nil }
        var cameraWarp: Homography?? = nil
        switch cameraKind {
        case 0: cameraWarp = nil
        case 1: cameraWarp = .some(nil)
        case 2:
            var m: [Double] = []
            for _ in 0..<9 { guard let bits = get(UInt64.self) else { return nil }; m.append(Double(bitPattern: bits)) }
            cameraWarp = .some(Homography(m))
        default: return nil
        }
        guard let nearCount = get(UInt8.self), let farCount = get(UInt8.self),
              Int(nearCount) <= PerceiveRules.MAX_PERSONS, Int(farCount) <= PerceiveRules.MAX_PERSONS else { return nil }
        var rows = [PassRow(), PassRow()]
        for (pass, count) in [Int(nearCount), Int(farCount)].enumerated() {
            for person in 0..<count {
                for i in 0..<4 { guard let v = float() else { return nil }; rows[pass].boxes[person * 4 + i] = v }
                guard let confidence = float() else { return nil }
                rows[pass].confidences[person] = confidence
                for i in 0..<51 { guard let v = float() else { return nil }; rows[pass].keypoints[person * 51 + i] = v }
            }
        }
        let sample = LoggedSample(sourceFrame: Int(frame), cameraWarp: cameraWarp,
                                  perceived: PerceivedSample(near: rows[0], far: rows[1]))
        return (sample, offset - start)
    }
}
