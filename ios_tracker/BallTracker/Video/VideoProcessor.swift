import AVFoundation
import Foundation

/// Result of a full offline pass over a video: one sample per decoded frame,
/// plus the rally segments the trace-driven detector cut from the clip.
struct VideoAnalysis {
    struct Sample {
        let t: Double
        let pos: CGPoint?    // tracked ball, source px (nil when no track)
        let state: TrackState
        let speedPxS: Double
    }

    let url: URL
    let size: CGSize     // display-oriented
    let fps: Double
    let duration: Double
    let samples: [Sample]
    let rallySegments: [RallySegment]
    let avgInferenceMs: Double
    /// Fenced stationary-clutter rects, in source (display) px, for the overlay.
    let exclusionZones: [CGRect]

    /// Fraction of frames with a live moving/coasting trace.
    var liveTraceRate: Double {
        guard !samples.isEmpty else { return 0 }
        let live = samples.filter { $0.state == .moving || $0.state == .coasting }.count
        return Double(live) / Double(samples.count)
    }

    var maxSpeedPxS: Double { samples.map(\.speedPxS).max() ?? 0 }

    /// Samples in `(t - window, t]`, for drawing the trail at playback time t.
    func trail(at t: Double, window: Double = 0.5) -> [CGPoint] {
        samples
            .filter { $0.t > t - window && $0.t <= t }
            .compactMap { s in
                (s.state == .moving || s.state == .coasting) ? s.pos : nil
            }
    }

    func sample(at t: Double) -> Sample? {
        // Samples are time-ordered; binary search for the last one <= t.
        var lo = 0, hi = samples.count - 1, best = -1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if samples[mid].t <= t { best = mid; lo = mid + 1 } else { hi = mid - 1 }
        }
        return best >= 0 ? samples[best] : nil
    }
}

enum VideoProcessorError: Error {
    case noVideoTrack
    case readFailed
}

/// Decodes a video with AVAssetReader and runs every frame through the ball
/// detector and YOLO player detector (Pass 1), then replays the detections
/// through the streaming Kalman/IMM tracker — dropping carried balls via the
/// carry suppressor — and the rally detector (Pass 2), the offline port of
/// pipeline/rally_detector.py's collect_rally_segments.
final class VideoProcessor {
    private let modelURL: URL?
    private let playerModelURL: URL?
    private let roiModelURL: URL?

    /// `modelURL` / `playerModelURL` override the app-bundle ball and player
    /// models (used by the test harness); nil falls back to the bundled models.
    /// `roiModelURL`, when set, enables the tracked-ROI + tiered-scan path with
    /// that small-input ball model.
    init(modelURL: URL? = nil, playerModelURL: URL? = nil, roiModelURL: URL? = nil) {
        self.modelURL = modelURL
        self.playerModelURL = playerModelURL
        self.roiModelURL = roiModelURL
    }

    /// Identifies the detector inputs a Pass-1 checkpoint depends on. If any of
    /// these change, a stored checkpoint's detections are stale and must not be
    /// resumed (rally post-processing is deliberately absent — it only affects
    /// Pass 2).
    static func detectorFingerprint(conf: Float, roi: Bool) -> String {
        "ball_best-v1|conf=\(conf)|aw=\(TrackerEngine.analysisWidth)|players=yolo26n|carry1"
            + (roi ? "|roi480p25" : "")
    }

    /// `checkpoint` is opt-in: when nil, this runs exactly as a single-shot pass
    /// (the parity/tuning harness path). When set, Pass-1 detections are flushed
    /// to disk periodically and an interrupted run resumes from the last flush.
    func process(url: URL,
                 conf: Float = BallDetector.defaultConf,
                 checkpoint: (store: CheckpointStore, key: String, name: String)? = nil,
                 progress: @escaping @Sendable (Double) -> Void) async throws -> VideoAnalysis {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoProcessorError.noVideoTrack
        }
        let duration = try await asset.load(.duration).seconds
        let fps = Double(try await track.load(.nominalFrameRate))

        // A video composition applies the track's preferredTransform, so
        // buffers come out display-oriented and overlay math matches AVPlayer.
        let composition = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
        let size = composition.renderSize

        let engine = try TrackerEngine(fps: fps > 0 ? fps : 30, conf: conf,
                                       modelURL: modelURL, roiModelURL: roiModelURL)
        let playerDetector = try playerModelURL.map {
            try PlayerDetector(analysisWidth: Double(TrackerEngine.analysisWidth), modelURL: $0)
        } ?? PlayerDetector(analysisWidth: Double(TrackerEngine.analysisWidth))
        let fingerprint = Self.detectorFingerprint(conf: conf, roi: roiModelURL != nil)

        // Pass 1 — detect every frame (ball candidates + player boxes). Unlike
        // the live path this commits to no associations: the whole clip's
        // evidence is gathered first, then replayed through the tracker.
        var perFrame: [[TrackerDetection]] = []
        var players: [PlayerBoxes] = []
        var times: [Double] = []
        var inferenceTotal = 0.0

        // Resume boundary: presentation time of the last frame already recorded
        // in a matching checkpoint. Frames at or before it are skipped so no
        // inference is repeated.
        var resumeBoundary = -Double.greatestFiniteMagnitude
        var record: ProcessingCheckpoint?

        if let checkpoint,
           let cp = checkpoint.store.load(key: checkpoint.key),
           cp.detectorFingerprint == fingerprint, !cp.completed {
            // Resume: restore Pass-1 state and the zones it was computed with,
            // so the scan pass is skipped and clutter is fenced off identically.
            perFrame = cp.frames
            players = cp.players
            times = cp.times
            inferenceTotal = cp.inferenceTotalMs
            resumeBoundary = cp.lastFrameTime
            engine.setExclusionZones(ExclusionZones(
                rects: cp.zones.map { CGRect(x: $0[0], y: $0[1],
                                             width: $0[2] - $0[0], height: $0[3] - $0[1]) }))
            record = cp
            if duration > 0 { progress(min(resumeBoundary / duration, 1.0) * 0.95) }
        } else {
            // Fence off stationary ball-like clutter first (ball baskets, balls
            // at rest). Without this a basket — detected on every frame — holds
            // the confirmed track forever and the real ball never gets promoted.
            try await engine.prepareExclusionZones(asset: asset, frameSize: size)
            if let checkpoint {
                record = ProcessingCheckpoint(
                    videoKey: checkpoint.key,
                    workingCopyName: url.lastPathComponent,
                    displayName: checkpoint.name,
                    detectorFingerprint: fingerprint,
                    fps: fps, width: size.width, height: size.height, duration: duration,
                    zones: engine.exclusionZones.rects.map {
                        [$0.minX, $0.minY, $0.maxX, $0.maxY] },
                    lastFrameTime: -1, frameCount: 0, inferenceTotalMs: 0,
                    times: [], frames: [], players: [], completed: false)
                try? checkpoint.store.save(record!)
            }
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderVideoCompositionOutput(
            videoTracks: [track],
            videoSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.videoComposition = composition
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw VideoProcessorError.readFailed }
        reader.add(output)
        // Seek decode past the frames already processed. The composition clamps
        // its first output to the range start, synthesizing a phantom frame
        // there that is not a real source frame — so we start mid-gap and, in
        // the loop, drop everything within ~¾ of a frame of the boundary. That
        // discards both the phantom and the last already-recorded frame, leaving
        // the next real frame onward. PTS stay on the composition timeline, so
        // the appended `times` line up exactly with an uninterrupted run.
        let frameDur = fps > 0 ? 1.0 / fps : 1.0 / 30.0
        let resumeSkipBelow = resumeBoundary + 0.75 * frameDur
        if resumeBoundary > -Double.greatestFiniteMagnitude {
            reader.timeRange = CMTimeRange(
                start: CMTime(seconds: resumeBoundary + 0.5 * frameDur, preferredTimescale: 600),
                duration: .positiveInfinity)
        }
        guard reader.startReading() else { throw VideoProcessorError.readFailed }

        var lastProgressT = -1.0
        var lastCheckpointT = resumeBoundary

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            try autoreleasepool {
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
                let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
                if t < resumeSkipBelow { return }

                let (dets, ms, _) = try engine.analysisDetections(pixelBuffer, at: t)
                inferenceTotal += ms
                perFrame.append(dets)
                players.append(playerDetector.update(pixelBuffer: pixelBuffer, t: t))
                times.append(t)

                if duration > 0, t - lastProgressT > 0.25 {
                    lastProgressT = t
                    progress(min(t / duration, 1.0) * 0.95)
                }

                if var cp = record, t - lastCheckpointT >= 2.0 {
                    lastCheckpointT = t
                    cp.times = times
                    cp.frames = perFrame
                    cp.players = players
                    cp.lastFrameTime = t
                    cp.frameCount = times.count
                    cp.inferenceTotalMs = inferenceTotal
                    record = cp
                    try? checkpoint?.store.save(cp)
                }
            }
        }

        if reader.status == .failed {
            throw reader.error ?? VideoProcessorError.readFailed
        }

        // Pass 2 — replay the recorded detections through the streaming
        // Kalman/IMM tracker while the rally accumulator watches the trace,
        // exactly as rally_detector.py's main loop does frame by frame.
        try Task.checkCancellation()

        // Analysis space -> source px for display. Derived from the composition
        // render size (which is the decoded buffer width) rather than the
        // engine's per-frame scale, so a resume that processes zero new frames
        // still maps positions correctly.
        let analysisScale = size.width > 0 ? TrackerEngine.analysisWidth / size.width : 1
        let toSource = analysisScale > 0 ? 1 / Double(analysisScale) : 1

        let manager = BallTrackManager(
            fps: fps > 0 ? fps : 30,
            perspectiveScale: makeImageRowPerspective(
                frameHeight: Double(size.height * analysisScale)))
        let rally = RallyAccumulator()
        // Drop carried balls (inside a moving player box, sharing its velocity)
        // before they reach the tracker — generalises the pipeline's near-player
        // carry test to every YOLO-detected player.
        let suppressor = CarrySuppressor(analysisWidth: Double(TrackerEngine.analysisWidth))
        var suppressedTotal = 0
        var bodyClutterTotal = 0
        var samples: [VideoAnalysis.Sample] = []
        samples.reserveCapacity(times.count)

        for i in times.indices {
            let framePlayers = i < players.count ? players[i] : .none
            let dets = suppressor.filter(perFrame[i], players: framePlayers, t: times[i])
            suppressedTotal += suppressor.lastSuppressedCount
            bodyClutterTotal += suppressor.lastBodyClutterCount
            let status = manager.update(detections: dets, now: times[i])
            rally.observe(status: status,
                          players: framePlayers,
                          t: times[i],
                          lastDetectionTime: manager.lastDetectionTime)
            samples.append(VideoAnalysis.Sample(
                t: times[i],
                pos: status.position.map {
                    CGPoint(x: $0.x * toSource, y: $0.y * toSource) },
                state: status.state,
                speedPxS: status.speedPxS * toSource))
        }
        let segments = rally.finish(videoDuration: duration)
        print("[CARRY] suppressed \(suppressedTotal) detection(s) "
              + "(\(bodyClutterTotal) body-clutter, \(suppressedTotal - bodyClutterTotal) carried)")
        if let p = engine.roiPlanner {
            let frames = p.roiFrames + p.scanFrames
            print(String(format: "[ROI] roi=%d scan=%d frames, %.2f inferences/frame",
                         p.roiFrames, p.scanFrames,
                         frames > 0 ? Double(p.tileInferences) / Double(frames) : 0))
        }
        progress(1.0)

        // Pass 1 is done and its detections are baked into the analysis; the
        // checkpoint has served its purpose. Mark it complete so the resume
        // prompt won't offer it again (pruning reclaims the working copy later).
        if let checkpoint { checkpoint.store.markCompleted(key: checkpoint.key) }

        // Zones are analysis-space rects; scale to source px for the overlay.
        let zonesSource = engine.exclusionZones.rects.map {
            CGRect(x: $0.minX * toSource, y: $0.minY * toSource,
                   width: $0.width * toSource, height: $0.height * toSource)
        }

        return VideoAnalysis(
            url: url,
            size: size,
            fps: fps,
            duration: duration,
            samples: samples,
            rallySegments: segments,
            avgInferenceMs: times.isEmpty ? 0 : inferenceTotal / Double(times.count),
            exclusionZones: zonesSource)
    }
}
