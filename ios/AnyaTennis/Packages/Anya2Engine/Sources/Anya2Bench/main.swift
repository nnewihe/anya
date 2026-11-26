// anya2-bench: how fast is the app's analysis, per sample, for each compute
// choice x the Core ML planning hints each iOS version offers?
//
//   anya2-bench MATCH_DIR MODEL.mlpackage [SEGMENTS=8] [ROUNDS=2]
//
// MATCH_DIR is an app match folder (manifest.json + proxies/), so the input
// is exactly what AnalysisScheduler sees: 15 Hz near (960x540) and far-band
// proxy segments. Per sample it times the two stages that run per frame:
//
//   decode    proxy NV12 -> BGRA (BT.601), near + far     (CPU, same on any OS)
//   perceive  letterbox + pose + decode + NMS, near + far (Perceiver.perceive)
//
// The camera track (every 3rd sample, OpenCV) and the anya2 engine are CPU
// code that no iOS version changes, so they are left out.
//
// Hint levels stand in for iOS versions on the same hardware:
//   iOS 16     no optimizationHints
//   iOS 17.4   reshapeFrequency = .infrequent
//   iOS 18     + specializationStrategy = .fastPrediction
// Each config runs ROUNDS times; the table reports the LAST round (OS model
// caches warm) and the first round's load time (cold-ish: includes any
// Neural Engine compile / specialization not already cached by the OS).
import Anya2Engine
import Anya2Perception
import AVFoundation
import CoreML
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    print("usage: anya2-bench MATCH_DIR MODEL.mlpackage [SEGMENTS=8] [ROUNDS=2]")
    exit(2)
}
let matchDir = URL(fileURLWithPath: arguments[1])
let segmentLimit = arguments.count > 3 ? Int(arguments[3])! : 8
let rounds = arguments.count > 4 ? Int(arguments[4])! : 2

// ── the match ─────────────────────────────────────────────────────────────
struct Manifest: Decodable {
    struct Band: Decodable { let x1, y1, x2, y2: Int }
    struct Segment: Decodable { let index, count: Int; let nearFile, farFile: String }
    let sourceWidth, sourceHeight: Int
    let band: Band
    let segments: [Segment]
}
let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: matchDir.appendingPathComponent("manifest.json")))
let band = FarBand(x1: manifest.band.x1, y1: manifest.band.y1, x2: manifest.band.x2, y2: manifest.band.y2,
                   scaleX: Double(manifest.sourceWidth) / Double(Court.ANALYSIS_WIDTH),
                   scaleY: Double(manifest.sourceHeight) / Double(Court.ANALYSIS_HEIGHT))

/// One proxy segment's frames, by pose sample index, timing the decode.
func decode(_ url: URL, seconds: inout Double) throws -> [Int: CVPixelBuffer] {
    let asset = AVURLAsset(url: url)
    let semaphore = DispatchSemaphore(value: 0)
    var track: AVAssetTrack?
    Task { track = try? await asset.loadTracks(withMediaType: .video).first; semaphore.signal() }
    semaphore.wait()
    guard let track else { throw CocoaError(.fileReadCorruptFile) }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    reader.startReading()
    var frames: [Int: CVPixelBuffer] = [:]
    while true {
        let start = DispatchTime.now().uptimeNanoseconds
        guard let sample = output.copyNextSampleBuffer() else { break }
        guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
        let bgra = bgraBT601(fromVideoRangeYUV: pixels)
        seconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let t = CMSampleBufferGetPresentationTimeStamp(sample).seconds
        if t.isFinite { frames[Int((t * 15).rounded())] = bgra }
    }
    return frames
}

var samples: [(near: CVPixelBuffer, far: CVPixelBuffer)] = []
var decodeSeconds = 0.0
for segment in manifest.segments.sorted(by: { $0.index < $1.index }).prefix(segmentLimit) {
    let near = try decode(matchDir.appendingPathComponent(segment.nearFile), seconds: &decodeSeconds)
    let far = try decode(matchDir.appendingPathComponent(segment.farFile), seconds: &decodeSeconds)
    for i in 0..<segment.count { if let n = near[i], let f = far[i] { samples.append((n, f)) } }
}
guard !samples.isEmpty else { print("no samples decoded"); exit(1) }
let decodeMs = decodeSeconds / Double(samples.count) * 1000
print(String(format: "%d samples from %d segments (far band %dx%d); decode + BT.601: %.2f ms/sample",
             samples.count, min(segmentLimit, manifest.segments.count), band.encodedWidth, band.encodedHeight, decodeMs))

// ── configurations ────────────────────────────────────────────────────────
enum HintLevel: String, CaseIterable { case iOS16 = "iOS 16 (no hints)", iOS17_4 = "iOS 17.4 hints", iOS18 = "iOS 18 hints" }
let units: [(String, MLComputeUnits)] = [("CPU", .cpuOnly), ("GPU", .cpuAndGPU),
                                         ("Neural Engine", .cpuAndNeuralEngine), ("all (Core ML picks)", .all)]

func configuration(_ computeUnits: MLComputeUnits, _ level: HintLevel) -> MLModelConfiguration? {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    switch level {
    case .iOS16: return configuration
    case .iOS17_4:
        guard #available(macOS 14.4, iOS 17.4, *) else { return nil }
        var hints = MLOptimizationHints()
        hints.reshapeFrequency = .infrequent
        configuration.optimizationHints = hints
    case .iOS18:
        guard #available(macOS 15.0, iOS 18.0, *) else { return nil }
        var hints = MLOptimizationHints()
        hints.reshapeFrequency = .infrequent
        hints.specializationStrategy = .fastPrediction
        configuration.optimizationHints = hints
    }
    return configuration
}

let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: arguments[2]))
print("model compiled; \(ComputeProfile.current)")

struct Result { let unit: String; let level: HintLevel; let firstLoadMs: Double; var ms: [Double] }
var results: [String: Result] = [:]
let warmup = 10
for round in 1...rounds {
    for (unitName, computeUnits) in units {
        for level in HintLevel.allCases {
            guard let configuration = configuration(computeUnits, level) else {
                print("skip \(unitName) / \(level.rawValue): not available on this OS"); continue
            }
            let key = unitName + level.rawValue
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let perceiver = Perceiver(nearModel: try PoseModel(compiledModelURL: compiled, configuration: configuration),
                                      farModel: try PoseModel(compiledModelURL: compiled, configuration: configuration),
                                      band: band)
            let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e6
            var times: [Double] = []
            for (i, sample) in samples.enumerated() {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try autoreleasepool { try perceiver.perceive(nearFrame: sample.near, farBandFrame: sample.far) }
                if i >= warmup { times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6) }
            }
            let mean = times.reduce(0, +) / Double(times.count)
            print(String(format: "round %d  %-20@ %-18@ load %7.0f ms   %6.2f ms/sample",
                         round, unitName as NSString, level.rawValue as NSString, loadMs, mean))
            if round == 1 { results[key] = Result(unit: unitName, level: level, firstLoadMs: loadMs, ms: times) }
            else { results[key]?.ms = times }
        }
    }
}

// ── table ─────────────────────────────────────────────────────────────────
func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * p))]
}
let samplesPerMinute = 60 * PerceiveRules.POSE_FPS
print("\n| Runs on | Hints | First load (ms) | Pose ms/sample (mean / p50 / p95) | Total ms/sample incl. decode | Analysis s per min recorded | x real time |")
print("|---|---|---|---|---|---|---|")
for (unitName, _) in units {
    for level in HintLevel.allCases {
        guard let r = results[unitName + level.rawValue] else { continue }
        let mean = r.ms.reduce(0, +) / Double(r.ms.count)
        let total = mean + decodeMs
        let perMinute = total * samplesPerMinute / 1000
        print(String(format: "| %@ | %@ | %.0f | %.2f / %.2f / %.2f | %.2f | %.1f | %.1fx |",
                     unitName, level.rawValue, r.firstLoadMs, mean, percentile(r.ms, 0.5), percentile(r.ms, 0.95),
                     total, perMinute, 60 / perMinute))
    }
}
