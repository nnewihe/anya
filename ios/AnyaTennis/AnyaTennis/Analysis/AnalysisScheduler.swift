// AnalysisScheduler.swift
//
// Runs anya2 on the match while it is being recorded, a segment at a time:
//
//   proxy segment (2 s) -> decode -> pose on near + far (Perceiver)
//     -> camera track on every 3rd sample -> LiveMatchAnalysis
//     -> perception.log (so a restart resumes here)
//
// It runs behind the camera by however long a segment takes to analyse, and
// catches up when it can. Recording always wins: when the phone gets hot
// (thermal state serious or worse) analysis pauses between segments and the
// backlog drains after Stop.
//
// Invariant: every sample handed to LiveMatchAnalysis is in the log, in
// order, with no holes -- each one is logged right after it is added. So a
// restore (replaying the log) is exactly where the live run had got to.
import Anya2Engine
import Anya2Perception
import AVFoundation
import Foundation

actor AnalysisScheduler {
    struct Status: Sendable {
        var samplesAnalysed = 0
        var secondsAnalysed = 0.0
        var segmentsWaiting = 0
        var nearPlayers = 0
        var farPlayers = 0
        var pausedForHeat = false
    }

    let matchID: String
    private let analysis: LiveMatchAnalysis
    private let perceiver: Perceiver
    private let log: PerceptionLog
    private let sourceFPS: Double
    private var queue: [MatchManifest.Segment] = []      // sorted by index, no repeats
    private var queuedIndices = Set<Int>()
    private var running = false
    private var ignoreHeat = false                       // set at Stop; the running loop reads it
    private var status = Status()
    private var onStatus: (@Sendable (Status) -> Void)?

    /// Restore from the perception log (cheap: no pose is re-run) or start
    /// fresh. Off the main thread: a long match's log takes seconds to replay.
    static func open(manifest: MatchManifest) async throws -> AnalysisScheduler {
        try await Task.detached(priority: .userInitiated) { try AnalysisScheduler(manifest: manifest) }.value
    }

    private init(manifest: MatchManifest) throws {
        matchID = manifest.id
        sourceFPS = manifest.sourceFPS
        let corners = manifest.courtCorners.map { (x: $0.x, y: $0.y) }
        let calibration = try CalibrationFrame.load(matchID: manifest.id)
        let reference = try calibration.withGray { try CameraReference(calibrationFrame: $0, courtCorners: corners) }
        let analysis = LiveMatchAnalysis(reference: reference, referenceHomography: Homography(courtCorners: corners),
                                         sourceFPS: manifest.sourceFPS, walkingModel: try WalkingModel.bundled())
        self.analysis = analysis
        let band = FarBand(x1: manifest.band.x1, y1: manifest.band.y1, x2: manifest.band.x2, y2: manifest.band.y2,
                           scaleX: Double(manifest.sourceWidth) / Double(Court.ANALYSIS_WIDTH),
                           scaleY: Double(manifest.sourceHeight) / Double(Court.ANALYSIS_HEIGHT))
        perceiver = try Perceiver(compiledModelURL: Self.modelURL(), band: band)
        log = PerceptionLog(url: MatchStore.file("perception.log", of: manifest.id))
        // Replay. Entries are in order by construction; anything else (an
        // older build's hole, a duplicate) is bridged with nobody-seen samples
        // or skipped rather than trusted.
        let empty = PerceivedSample(near: PassRow(), far: PassRow())
        log.readEach { entry in
            guard entry.sourceFrame >= analysis.expectedSourceFrame,
                  (entry.sourceFrame - analysis.expectedSourceFrame) % analysis.poseStride == 0 else { return }
            while analysis.expectedSourceFrame < entry.sourceFrame {
                let frame = analysis.expectedSourceFrame
                analysis.replay(empty, sourceFrame: frame,
                                cameraWarp: analysis.camera.isSampleFrame(frame) ? .some(nil) : nil)
            }
            analysis.replay(entry.perceived, sourceFrame: entry.sourceFrame, cameraWarp: entry.cameraWarp)
        }
        status = Status()
        status.samplesAnalysed = analysis.expectedSourceFrame / analysis.poseStride
        status.secondsAnalysed = Double(analysis.expectedSourceFrame) / sourceFPS
        for segment in manifest.segments { insert(segment) }
    }

    static func modelURL() throws -> URL {
        guard let url = Bundle.main.url(forResource: "PoseModel", withExtension: "mlmodelc") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return url
    }

    func setStatusHandler(_ handler: @escaping @Sendable (Status) -> Void) {
        onStatus = handler
        handler(status)
    }

    /// A new segment is on disk.
    func enqueue(_ segment: MatchManifest.Segment) {
        insert(segment)
        publish()
        if !running { Task { await drain() } }
    }

    /// Queue `segment` unless it is queued already or the log covers it,
    /// keeping the queue in timeline order (segments can finish writing out
    /// of order).
    private func insert(_ segment: MatchManifest.Segment) {
        guard !queuedIndices.contains(segment.index),
              segment.firstSourceFrame + segment.count * analysis.poseStride > analysis.expectedSourceFrame else { return }
        queuedIndices.insert(segment.index)
        let at = queue.firstIndex { $0.index > segment.index } ?? queue.count
        queue.insert(segment, at: at)
    }

    /// Analyse everything queued (one loop at a time).
    func drain() async {
        guard !running else { return }
        running = true
        defer { running = false }
        while let segment = queue.first {
            if !ignoreHeat, ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                status.pausedForHeat = true
                publish()
                try? await Task.sleep(for: .seconds(5))
                continue
            }
            status.pausedForHeat = false
            do { try await analyse(segment) } catch {
                // An unreadable segment (e.g. cut short by a kill) or a pose
                // failure counts as nobody seen: the timeline must not stall.
                try? fillGap(upTo: segment.firstSourceFrame + segment.count * analysis.poseStride)
            }
            queue.removeFirst()
            queuedIndices.remove(segment.index)
            publish()
        }
    }

    /// The whole match's result, once every recorded segment is analysed.
    /// `segments`: the manifest's, so nothing still in flight is missed.
    func finish(segments: [MatchManifest.Segment]) async -> BatchResult {
        ignoreHeat = true
        for segment in segments { insert(segment) }
        while running || !queue.isEmpty {
            if !running { await drain() } else { try? await Task.sleep(for: .milliseconds(100)) }
        }
        return analysis.finish()
    }

    // ── one segment ───────────────────────────────────────────────────────

    private func analyse(_ segment: MatchManifest.Segment) async throws {
        try fillGap(upTo: segment.firstSourceFrame)
        let folder = MatchStore.folder(matchID)
        let near = try await ProxySegmentReader(url: folder.appendingPathComponent(segment.nearFile))
        let far = try await ProxySegmentReader(url: folder.appendingPathComponent(segment.farFile))
        let empty = PerceivedSample(near: PassRow(), far: PassRow())
        var nearFrame = near.next(), farFrame = far.next()
        for i in 0..<segment.count {
            let sourceFrame = segment.firstSourceFrame + i * analysis.poseStride
            // Frames are matched to samples by timestamp, so a frame the
            // encoder dropped leaves one sample empty instead of shifting
            // every later one.
            while let frame = nearFrame, frame.sample < i { nearFrame = near.next() }
            while let frame = farFrame, frame.sample < i { farFrame = far.next() }
            guard sourceFrame >= analysis.expectedSourceFrame else { continue }   // already in the log
            var perceived = empty
            if let n = nearFrame, n.sample == i, let f = farFrame, f.sample == i {
                perceived = try autoreleasepool { try perceiver.perceive(nearFrame: n.pixels, farBandFrame: f.pixels) }
            }
            var cameraWarp: Homography?? = nil
            if analysis.camera.isSampleFrame(sourceFrame), let n = nearFrame, n.sample == i {
                let gray = grayscale(of: n.pixels)
                gray.pixels.withUnsafeBufferPointer { buffer in
                    analysis.add(perceived, sourceFrame: sourceFrame,
                                 cameraFrame: GrayFrame(pixels: buffer.baseAddress!, width: gray.width,
                                                        height: gray.height, bytesPerRow: gray.width))
                }
                cameraWarp = .some(analysis.camera.lastRegisteredWarp)
            } else {
                analysis.add(perceived, sourceFrame: sourceFrame, cameraFrame: nil)
                if analysis.camera.isSampleFrame(sourceFrame) { cameraWarp = .some(nil) }
            }
            try log.append([LoggedSample(sourceFrame: sourceFrame, cameraWarp: cameraWarp, perceived: perceived)],
                           sync: false)
            status.nearPlayers = perceived.near.confidences.filter { !$0.isNaN }.count
            status.farPlayers = perceived.far.confidences.filter { !$0.isNaN }.count
        }
        // A segment cut short (kill mid-write) leaves its tail samples empty.
        try fillGap(upTo: segment.firstSourceFrame + segment.count * analysis.poseStride)
        try log.append([], sync: true)
    }

    /// Nobody-seen samples for timeline frames no segment covers (the seam
    /// between two recording parts, or a lost segment), each logged as added.
    private func fillGap(upTo sourceFrame: Int) throws {
        let empty = PerceivedSample(near: PassRow(), far: PassRow())
        while analysis.expectedSourceFrame < sourceFrame {
            let frame = analysis.expectedSourceFrame
            analysis.add(empty, sourceFrame: frame, cameraFrame: nil)
            try log.append([LoggedSample(sourceFrame: frame,
                                         cameraWarp: analysis.camera.isSampleFrame(frame) ? .some(nil) : nil,
                                         perceived: empty)], sync: false)
        }
    }

    private func publish() {
        status.samplesAnalysed = analysis.expectedSourceFrame / analysis.poseStride
        status.secondsAnalysed = Double(analysis.expectedSourceFrame) / sourceFPS
        status.segmentsWaiting = queue.count
        onStatus?(status)
    }
}

/// Decodes one proxy segment to BGRA frames, each with the pose sample it
/// belongs to (from its timestamp: ProxyRecorder writes sample k at k/15 s).
final class ProxySegmentReader {
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput

    init(url: URL) async throws {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw CocoaError(.fileReadCorruptFile)
        }
        reader = try AVAssetReader(asset: asset)
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
    }

    func next() -> (sample: Int, pixels: CVPixelBuffer)? {
        while let sample = output.copyNextSampleBuffer() {
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            let seconds = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard seconds.isFinite else { continue }
            return (Int((seconds * 15).rounded()), pixels)
        }
        return nil
    }
}
