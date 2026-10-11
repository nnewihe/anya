// MatchController.swift
//
// One match from calibration to reel. Owns the recorders and the analysis
// while the match is open, keeps the manifest saved at every step, and
// publishes what the screens show.
//
//   calibrate(corners)   save the calibration frame, compute the far band
//   startRecording()     begin a part (the first, or after an interruption)
//   pauseRecording()     end the part (phone call, app backgrounded, kill)
//   stop()               end the part, finish analysis, cut the reel
//
// The camera thread only touches the recorders; manifest changes funnel
// through the main actor.
import Anya2Engine
import Anya2Perception
import AVFoundation
import Observation
import UIKit

@MainActor
@Observable
final class MatchController {
    private(set) var manifest: MatchManifest
    private(set) var analysisStatus = AnalysisScheduler.Status()
    private(set) var recordingSince: Date?
    private(set) var finishingStep = ""
    private(set) var error: String?

    let camera: CameraSession
    private var scheduler: AnalysisScheduler?
    private var master: MasterRecorder?
    private var proxies: ProxyRecorder?
    private let frameCounter = FrameCounter()
    private var endingPart = false
    private var schedulerOpening: Task<AnalysisScheduler, Error>?

    init(manifest: MatchManifest, camera: CameraSession) {
        self.manifest = manifest
        self.camera = camera
    }

    var isRecording: Bool { recordingSince != nil }

    /// Seconds recorded across all parts (the live part included).
    var secondsRecorded: Double {
        let finished = Double(manifest.parts.dropLast().reduce(0) { $0 + $1.frameCount }) / manifest.sourceFPS
        guard let last = manifest.parts.last else { return 0 }
        if let recordingSince { return finished + Date().timeIntervalSince(recordingSince) }
        return finished + Double(last.frameCount) / manifest.sourceFPS
    }

    // ── a new match ───────────────────────────────────────────────────────

    /// Create the match from the four tapped corners (analysis px) and the
    /// 960x540 frame they were tapped on.
    static func create(corners: [CGPoint], calibrationFrame: CVPixelBuffer, camera: CameraSession,
                       sourceWidth: Int = CameraSession.width, sourceHeight: Int = CameraSession.height,
                       sourceFPS: Double = CameraSession.fps) throws -> MatchController {
        let points = corners.map { (x: Double($0.x), y: Double($0.y)) }
        let band = FarBand(imageToCourt: Homography(courtCorners: points),
                           sourceWidth: sourceWidth, sourceHeight: sourceHeight)
        guard band.width > 32, band.height > 32 else { throw SetupError.farBaselineOffScreen }
        let manifest = MatchManifest(
            id: UUID().uuidString, createdAt: Date(), state: .calibrated,
            sourceWidth: sourceWidth, sourceHeight: sourceHeight, sourceFPS: sourceFPS,
            courtCorners: points.map { .init(x: $0.x, y: $0.y) },
            band: .init(x1: band.x1, y1: band.y1, x2: band.x2, y2: band.y2))
        try MatchStore.create(manifest)
        try CalibrationFrame.save(calibrationFrame, matchID: manifest.id)
        return MatchController(manifest: manifest, camera: camera)
    }

    enum SetupError: LocalizedError {
        case farBaselineOffScreen, lowStorage, recordingFailed(String)
        var errorDescription: String? {
            switch self {
            case .farBaselineOffScreen: "The far baseline isn't in the picture. Aim the camera so the whole court shows."
            case .lowStorage: "Less than 10 GB free. A match needs about 15 GB an hour."
            case .recordingFailed(let why): "Recording stopped: \(why). What was recorded before is safe."
            }
        }
    }

    // ── recording ─────────────────────────────────────────────────────────

    func startRecording() async throws {
        guard !isRecording, !endingPart else { return }
        #if !targetEnvironment(simulator)
        guard MatchStore.freeBytes() > 10_000_000_000 else { throw SetupError.lowStorage }
        #endif
        _ = try await analysisScheduler()   // restores from the log before new segments arrive
        let stride = PerceiveRules.stride(sourceFPS: manifest.sourceFPS).stride
        let cameraStep = CameraRules.sampleStep(sourceFPS: manifest.sourceFPS)
        let firstFrame = Self.nextPartStart(after: manifest.parts, grid: cameraStep * stride / gcd(cameraStep, stride))
        let file = String(format: "part-%03d.mov", manifest.parts.count + 1)
        let master = try MasterRecorder(url: MatchStore.file(file, of: manifest.id), width: manifest.sourceWidth,
                                        height: manifest.sourceHeight,
                                        withAudio: AVCaptureDevice.authorizationStatus(for: .audio) == .authorized)
        let band = FarBand(x1: manifest.band.x1, y1: manifest.band.y1, x2: manifest.band.x2, y2: manifest.band.y2,
                           scaleX: Double(manifest.sourceWidth) / Double(Court.ANALYSIS_WIDTH),
                           scaleY: Double(manifest.sourceHeight) / Double(Court.ANALYSIS_HEIGHT))
        let scaler = FrameScaler(band: .init(x: band.x1, y: band.y1, width: band.encodedWidth, height: band.encodedHeight))
        let proxies = ProxyRecorder(folder: MatchStore.folder(manifest.id), scaler: scaler,
                                    sourceFPS: manifest.sourceFPS, poseStride: stride, firstSourceFrame: firstFrame,
                                    firstSegmentIndex: (manifest.segments.map(\.index).max() ?? 0) + 1)
        proxies.onSegment = { [weak self] segment in await self?.segmentFinished(segment) }
        self.master = master
        self.proxies = proxies
        manifest.parts.append(.init(file: file, firstSourceFrame: firstFrame, frameCount: 0, startedAt: Date()))
        manifest.state = .recording
        save()
        UIApplication.shared.isIdleTimerDisabled = true
        recordingSince = Date()

        let counter = frameCounter
        counter.set(-1)
        var reported = false
        camera.setHandlers(video: { [weak self] sample in
            let frame = proxies.sourceFrame(of: sample)
            master.append(video: sample)
            proxies.append(sample, sourceFrame: frame)
            counter.set(frame - firstFrame)
            // Watchdog: a failed writer or a nearly full disk ends the part
            // (with a message) instead of recording nothing while the clock runs.
            guard !reported, frame % 30 == 0 else { return }
            let problem: String? = !master.isHealthy || !proxies.isHealthy
                ? (master.error?.localizedDescription ?? "the video writer failed")
                : MatchStore.freeBytes() < 1_500_000_000 ? "the phone is out of storage" : nil
            if let problem {
                reported = true
                Task { @MainActor in await self?.recordingFailed(problem) }
            }
        }, audio: { sample in master.append(audio: sample) })
    }

    /// Called by the watchdog: the part so far is kept, the match pauses.
    var onRecordingFailed: ((String) -> Void)?

    private func recordingFailed(_ why: String) async {
        await pauseRecording()
        error = SetupError.recordingFailed(why).localizedDescription
        onRecordingFailed?(why)
    }

    /// End the current part. The match stays open: recording can resume.
    func pauseRecording() async {
        let background = UIApplication.shared.beginBackgroundTask(withName: "pause recording")
        defer { UIApplication.shared.endBackgroundTask(background) }
        await endPart()
        if manifest.state == .recording { manifest.state = .paused }
        save()
    }

    /// Stop for good: finish analysis, then cut the reel.
    func stop() async {
        let background = UIApplication.shared.beginBackgroundTask(withName: "finish match")
        defer { UIApplication.shared.endBackgroundTask(background) }
        await endPart()
        manifest.state = .finishing
        save()
        await finish()
    }

    /// Analysis + reel for a stopped match (also retries a failed finish).
    func finish() async {
        do {
            finishingStep = "Analysing the last of the recording…"
            let scheduler = try await analysisScheduler()
            let result = await scheduler.finish(segments: manifest.segments)
            finishingStep = "Cutting the reel…"
            let segments = result.reel.segments.map { (start: $0.start, end: $0.stop) }
            guard !segments.isEmpty else {
                // Not an error to retry: nothing in the recording looked like a point.
                manifest.reel = nil
                manifest.state = .finished
                manifest.lastError = "No points were found in this recording."
                finishingStep = ""
                save()
                return
            }
            let reelFile = "reel.mov"
            try await ReelExporter.export(manifest: manifest, segments: segments,
                                          to: MatchStore.file(reelFile, of: manifest.id))
            manifest.reel = .init(file: reelFile, points: segments.count,
                                  seconds: segments.reduce(0) { $0 + $1.end - $1.start },
                                  nearServes: result.nearServes.count, farServes: result.farServes.count,
                                  pointEnds: result.pointEnds.count)
            manifest.state = .finished
            manifest.lastError = nil
            #if !DEBUG
            try? FileManager.default.removeItem(at: MatchStore.folder(manifest.id).appendingPathComponent("proxies"))
            #endif
        } catch {
            manifest.state = .failed
            manifest.lastError = error.localizedDescription
            #if DEBUG
            print("[finish] failed: \(String(reflecting: error))")
            #endif
            self.error = error.localizedDescription
        }
        finishingStep = ""
        save()
    }

    func updateInfo(_ info: MatchManifest.Info) {
        manifest.info = info
        save()
    }

    func update(_ change: (inout MatchManifest) -> Void) {
        change(&manifest)
        save()
    }

    // ── internals ─────────────────────────────────────────────────────────

    private func endPart() async {
        // Once only: Stop and an interruption can both arrive while the
        // writers are finishing.
        guard isRecording, !endingPart else { return }
        endingPart = true
        defer { endingPart = false }
        camera.setHandlers(video: nil, audio: nil)   // returns after any frame in flight
        await master?.finish()
        await proxies?.finish()
        let frames = frameCounter.value + 1
        if !manifest.parts.isEmpty { manifest.parts[manifest.parts.count - 1].frameCount = frames }
        master = nil
        proxies = nil
        recordingSince = nil
        UIApplication.shared.isIdleTimerDisabled = false
        save()
    }

    private func segmentFinished(_ segment: MatchManifest.Segment) async {
        manifest.segments.append(segment)
        manifest.segments.sort { $0.index < $1.index }
        save()
        await scheduler?.enqueue(segment)
    }

    /// The match's analysis, restored from its log the first time (off the
    /// main thread; concurrent callers share one restore).
    private func analysisScheduler() async throws -> AnalysisScheduler {
        if let scheduler { return scheduler }
        let opening = schedulerOpening ?? Task { [manifest] in try await AnalysisScheduler.open(manifest: manifest) }
        schedulerOpening = opening
        let made = try await opening.value
        if scheduler == nil {
            scheduler = made
            await made.setStatusHandler { status in
                Task { @MainActor [weak self] in
                    self?.analysisStatus = status
                    self?.manifest.samplesAnalysed = status.samplesAnalysed
                }
            }
            Task { await made.drain() }
        }
        return scheduler!
    }

    private func save() {
        do { try MatchStore.save(manifest) } catch { self.error = "Could not save the match: \(error.localizedDescription)" }
    }

    /// Where the next part starts on the timeline: after the previous one, on
    /// the grid where pose and camera samples coincide.
    static func nextPartStart(after parts: [MatchManifest.Part], grid: Int) -> Int {
        guard let last = parts.last else { return 0 }
        let end = last.firstSourceFrame + last.frameCount
        return (end + grid - 1) / grid * grid
    }

    // ── after a kill ──────────────────────────────────────────────────────

    /// A match the app died while recording: settle its last part from what
    /// is on disk and mark it paused, so it can be resumed or finished.
    static func recoverIfInterrupted(_ manifest: inout MatchManifest) async {
        guard manifest.state == .recording || manifest.state == .finishing else { return }
        if manifest.state == .recording, var last = manifest.parts.last, last.frameCount == 0 {
            let asset = AVURLAsset(url: MatchStore.file(last.file, of: manifest.id))
            let seconds = (try? await asset.load(.duration).seconds) ?? 0
            let frames = seconds * manifest.sourceFPS
            let inMaster = frames.isFinite && frames >= 0 && frames < 1e9 ? Int(frames.rounded()) : 0
            // A proxy segment can reach past the master's last fragment; the
            // part must cover both, or the next part would overlap analysed frames.
            let stride = PerceiveRules.stride(sourceFPS: manifest.sourceFPS).stride
            let inProxies = manifest.segments.filter { $0.firstSourceFrame >= last.firstSourceFrame }
                .map { $0.firstSourceFrame + $0.count * stride - last.firstSourceFrame }.max() ?? 0
            last.frameCount = max(inMaster, inProxies)
            manifest.parts[manifest.parts.count - 1] = last
            manifest.state = .paused
        } else if manifest.state == .finishing {
            manifest.state = .failed
            manifest.lastError = "The app closed while the reel was being made. Tap Finish to try again."
        }
        // Proxy files of a segment that never finished are unreadable; drop them.
        let known = Set(manifest.segments.flatMap { [$0.nearFile, $0.farFile] })
        let proxies = MatchStore.folder(manifest.id).appendingPathComponent("proxies")
        for name in (try? FileManager.default.contentsOfDirectory(atPath: proxies.path)) ?? []
        where !known.contains("proxies/" + name) {
            try? FileManager.default.removeItem(at: proxies.appendingPathComponent(name))
        }
        try? MatchStore.save(manifest)
    }
}

/// The last frame index the camera thread wrote, read by the main actor.
final class FrameCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var last = -1
    func set(_ value: Int) { lock.withLock { last = value } }
    var value: Int { lock.withLock { last } }
}

private func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
