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
    private var queue: [MatchManifest.Segment] = []
    private var running = false
    private var status: Status
    private var onStatus: (@Sendable (Status) -> Void)?

    /// Restore from the perception log (cheap: no pose is re-run) or start fresh.
    init(manifest: MatchManifest) throws {
        matchID = manifest.id
        sourceFPS = manifest.sourceFPS
        let corners = manifest.courtCorners.map { (x: $0.x, y: $0.y) }
        let calibration = try CalibrationFrame.load(matchID: manifest.id)
        let reference = try calibration.withGray { try CameraReference(calibrationFrame: $0, courtCorners: corners) }
        let homography = Homography(courtCorners: corners)
        let analysis = LiveMatchAnalysis(reference: reference, referenceHomography: homography,
                                         sourceFPS: manifest.sourceFPS, walkingModel: try WalkingModel.bundled())
        self.analysis = analysis
        let band = FarBand(x1: manifest.band.x1, y1: manifest.band.y1, x2: manifest.band.x2, y2: manifest.band.y2,
                           scaleX: Double(manifest.sourceWidth) / Double(Court.ANALYSIS_WIDTH),
                           scaleY: Double(manifest.sourceHeight) / Double(Court.ANALYSIS_HEIGHT))
        perceiver = try Perceiver(compiledModelURL: Self.modelURL(), band: band)
        let log = PerceptionLog(url: MatchStore.file("perception.log", of: manifest.id))
        self.log = log
        for entry in log.readAll().samples {
            analysis.replay(entry.perceived, sourceFrame: entry.sourceFrame, cameraWarp: entry.cameraWarp)
        }
        var status = Status()
        status.samplesAnalysed = analysis.expectedSourceFrame / analysis.poseStride
        self.status = status
        // Queue every finished segment the log does not fully cover yet.
        queue = manifest.segments.filter { $0.firstSourceFrame + $0.count * analysis.poseStride > analysis.expectedSourceFrame }
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
        queue.append(segment)
        status.segmentsWaiting = queue.count
        onStatus?(status)
        if !running { Task { await drain(ignoreHeat: false) } }
    }

    /// Analyse everything queued. After Stop, `ignoreHeat` drains regardless.
    func drain(ignoreHeat: Bool) async {
        guard !running else {
            while running { try? await Task.sleep(for: .milliseconds(200)) }
            if !queue.isEmpty { await drain(ignoreHeat: ignoreHeat) }
            return
        }
        running = true
        defer { running = false }
        while let segment = queue.first {
            if !ignoreHeat, ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
                status.pausedForHeat = true
                onStatus?(status)
                try? await Task.sleep(for: .seconds(5))
                continue
            }
            status.pausedForHeat = false
            do { try await analyse(segment) } catch {
                // An unreadable segment (e.g. cut short by a kill) counts as
                // nobody seen: the timeline must not stall on it.
                skip(segment)
            }
            queue.removeFirst()
            status.segmentsWaiting = queue.count
            onStatus?(status)
        }
    }

    /// The whole match's result, once everything recorded has been analysed.
    func finish() async -> BatchResult {
        await drain(ignoreHeat: true)
        return analysis.finish()
    }

    // ── one segment ───────────────────────────────────────────────────────

    private func analyse(_ segment: MatchManifest.Segment) async throws {
        fillGap(upTo: segment.firstSourceFrame)
        let folder = MatchStore.folder(matchID)
        let near = try await ProxySegmentReader(url: folder.appendingPathComponent(segment.nearFile))
        let far = try await ProxySegmentReader(url: folder.appendingPathComponent(segment.farFile))
        var entries: [LoggedSample] = []
        for i in 0..<segment.count {
            let sourceFrame = segment.firstSourceFrame + i * analysis.poseStride
            guard let nearFrame = near.next(), let farFrame = far.next() else { break }
            guard sourceFrame >= analysis.expectedSourceFrame else { continue }   // already in the log
            let perceived = try autoreleasepool { try perceiver.perceive(nearFrame: nearFrame, farBandFrame: farFrame) }
            var cameraWarp: Homography?? = nil
            if analysis.camera.isSampleFrame(sourceFrame) {
                let gray = grayscale(of: nearFrame)
                gray.pixels.withUnsafeBufferPointer { buffer in
                    analysis.add(perceived, sourceFrame: sourceFrame,
                                 cameraFrame: GrayFrame(pixels: buffer.baseAddress!, width: gray.width,
                                                        height: gray.height, bytesPerRow: gray.width))
                }
                cameraWarp = .some(analysis.camera.lastRegisteredWarp)
            } else {
                analysis.add(perceived, sourceFrame: sourceFrame, cameraFrame: nil)
            }
            entries.append(LoggedSample(sourceFrame: sourceFrame, cameraWarp: cameraWarp, perceived: perceived))
            updatePlayers(perceived)
        }
        try log.append(entries)
        // A segment cut short (kill mid-write) leaves its tail samples empty.
        fillGap(upTo: segment.firstSourceFrame + segment.count * analysis.poseStride)
    }

    private func skip(_ segment: MatchManifest.Segment) {
        fillGap(upTo: segment.firstSourceFrame + segment.count * analysis.poseStride)
    }

    /// Samples with nobody in them for timeline frames no segment covers
    /// (the seam between two recording parts, or a lost segment).
    private func fillGap(upTo sourceFrame: Int) {
        var entries: [LoggedSample] = []
        while analysis.expectedSourceFrame < sourceFrame {
            let frame = analysis.expectedSourceFrame
            let empty = PerceivedSample(near: PassRow(), far: PassRow())
            analysis.add(empty, sourceFrame: frame, cameraFrame: nil)
            entries.append(LoggedSample(sourceFrame: frame,
                                        cameraWarp: analysis.camera.isSampleFrame(frame) ? .some(nil) : nil,
                                        perceived: empty))
        }
        if !entries.isEmpty { try? log.append(entries) }
    }

    private func updatePlayers(_ sample: PerceivedSample) {
        var status = Status()
        status.samplesAnalysed = analysis.expectedSourceFrame / analysis.poseStride
        self.status = status
        status.secondsAnalysed = Double(analysis.expectedSourceFrame) / sourceFPS
        status.nearPlayers = sample.near.confidences.filter { !$0.isNaN }.count
        status.farPlayers = sample.far.confidences.filter { !$0.isNaN }.count
    }
}

/// Decodes one proxy segment to BGRA frames.
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

    func next() -> CVPixelBuffer? {
        guard let sample = output.copyNextSampleBuffer() else { return nil }
        return CMSampleBufferGetImageBuffer(sample)
    }
}
