// LiveAnalysis.swift
//
// anya2 stages 5-8 while the match is still being recorded.
//
// Pose samples arrive in chunks. Each per-sample signal only looks a bounded
// distance into the future, so a sample's value is FINAL once enough later
// samples exist; from then on it is identical to what the batch run computes
// on the whole clip. Live analysis therefore:
//
//   - builds the player tracks sample by sample (they only look back);
//   - recomputes each per-sample signal over a window padded by its
//     dependency radius, and COMMITS only the samples that are now final
//     (committed values never change afterwards);
//   - leaves to Stop only what genuinely needs the whole clip: which near slot
//     leads the end signals (whole-clip coverage), the live score's
//     normalisation (the clip's 90th percentile), turning per-sample features
//     into events, and the orchestrator. All of that takes milliseconds.
//
// `finish()` returns exactly what `runBatch` returns on the same detections;
// LiveAnalysisTests checks that on every corpus clip, with random chunking.
import Foundation

/// How far into the future each signal looks, in pose samples at 15 Hz.
/// A sample is committed once this many later samples exist.
public enum LiveLookahead {
    /// Every per-sample feature: the walking classifier's +/-60-sample (8 s)
    /// window plus ~12 samples of gap filling, smoothing and differencing in
    /// its frame signals. near_end (<= 30), the serve features (<= 20) and
    /// player activity (9) all fit inside it.
    public static let FEATURES = 80
    /// The live score's 4 s smoothing reads 29 samples ahead of the features.
    public static let LIVE_SMOOTHING = 30

    /// Seconds between a pose sample arriving and its live score being final.
    public static func lagSeconds(fps: Double) -> Double { Double(FEATURES + LIVE_SMOOTHING) / fps }
}

public final class LiveAnalysis {
    public let fps: Double
    public let stride: Int
    private let geometry: Geometry
    private let walkingModel: WalkingModel
    private let reelConfig: ReelConfig
    private var trackBuilder: TrackBuilder

    // Committed per-sample signals. Index = pose sample.
    private var featuresCommitted = 0
    private(set) var nearServe: [Int: NearServeFeatures] = [:]         // by near slot
    private var farServe: [Int: FarServeFeatures] = [:]           // by far slot
    private var activity = [[Double]](repeating: [], count: Slots.count)
    // The end signals depend on which near slot leads, which is known only at
    // Stop, so both variants are kept. Index = lead slot (0 or 1).
    private(set) var walkingProbability: [[Double]] = [[], []]
    private var nearEnd: [[String: [Double]]] = [[:], [:]]
    private var liveCommitted = 0
    private(set) var smoothedLive: [[Double]] = [[], []]

    public init(fps: Double, stride: Int, geometry: Geometry, walkingModel: WalkingModel,
                reelConfig: ReelConfig = ReelConfig()) {
        self.fps = fps; self.stride = stride; self.geometry = geometry
        self.walkingModel = walkingModel; self.reelConfig = reelConfig
        trackBuilder = TrackBuilder(fps: fps, stride: stride, geometry: geometry)
        for slot in Slots.near { nearServe[slot] = .empty(fps: fps) }
        for slot in Slots.far { farServe[slot] = .empty(fps: fps) }
        for lead in 0..<2 {
            nearEnd[lead] = Dictionary(uniqueKeysWithValues: NearEndRules.SIGNAL_NAMES.map { ($0, []) })
        }
    }

    /// Pose samples received so far.
    public var samplesReceived: Int { trackBuilder.tracks.samples }
    /// Pose samples whose live score is final.
    public var samplesFinal: Int { liveCommitted }

    /// Add the next pose samples (the near and far passes for the same samples).
    public func append(_ samples: [PooledSample]) {
        for sample in samples { trackBuilder.add(sample) }
        advance(finished: false)
    }

    /// The match is over: commit the tail and run the whole-clip steps.
    public func finish() -> BatchResult {
        advance(finished: true)
        let tracks = trackBuilder.tracks
        let lead = leadNearSlot(tracks)
        let endSignals = EndSignals(fps: fps, walkingProbability: walkingProbability[lead], nearEnd: nearEnd[lead])
        let nearServes = keepOnePerWindow(
            Slots.near.filter(tracks.wasTracked(slot:)).flatMap { detectNearServes(nearServe[$0]!, track: $0) },
            gap: NearServeRules.REFRACT_S, time: { $0.t }, score: { $0.p })
            .map { Event(t: $0.t, p: $0.p, kind: .nearServe, track: $0.track, detail: $0.detail) }
        let farServes = keepOnePerWindow(
            Slots.far.filter(tracks.wasTracked(slot:)).flatMap { detectFarServes(farServe[$0]!, track: $0) },
            gap: FarServeRules.REFRACT_S, time: { $0.t }, score: { $0.p })
            .map { Event(t: $0.t, p: $0.p, kind: .farServe, track: $0.track, detail: $0.detail) }
        let live = normaliseLiveScore(smoothedLive[lead])
        let pointEnds = detectPointEnds(live: live, fps: fps)
        let reel = buildReel(near: nearServes, far: farServes, ends: pointEnds, live: live,
                             liveFPS: fps, tracks: tracks, config: reelConfig)
        return BatchResult(tracks: tracks, endSignals: endSignals, nearServes: nearServes, farServes: farServes,
                           pointEnds: pointEnds, live: live, reel: reel)
    }

    // MARK: - Committing

    private func advance(finished: Bool) {
        let available = trackBuilder.tracks.samples
        commitFeatures(through: finished ? available : max(featuresCommitted, available - LiveLookahead.FEATURES),
                       available: available)
        commitLiveScore(through: finished ? featuresCommitted
                        : max(liveCommitted, featuresCommitted - LiveLookahead.LIVE_SMOOTHING),
                        finished: finished)
    }

    /// Compute every per-sample feature over [commit start - FEATURES, available)
    /// and keep the rows [featuresCommitted, end).
    private func commitFeatures(through end: Int, available: Int) {
        guard end > featuresCommitted else { return }
        let windowStart = max(0, featuresCommitted - LiveLookahead.FEATURES)
        let window = trackBuilder.tracks.slice(windowStart..<available)
        let rows = (featuresCommitted - windowStart)..<(end - windowStart)

        for slot in Slots.near {
            let features = nearServeFeatures(keypoints: window.keypoints(slot: slot), boxes: window.boxes(slot: slot),
                                             fps: fps, courtY: window.court(slot: slot, axis: 1),
                                             eligible: window.eligible(slot: slot))
            nearServe[slot]!.appendRows(of: features, rows)
        }
        for slot in Slots.far {
            let features = farServeFeatures(keypoints: window.keypoints(slot: slot), boxes: window.boxes(slot: slot),
                                            fps: fps, eligible: window.eligible(slot: slot))
            farServe[slot]!.appendRows(of: features, rows)
        }
        let windowActivity = playerActivity(window)
        for slot in 0..<Slots.count { activity[slot] += windowActivity[slot][rows] }

        for lead in 0..<2 {
            let pose = nearPlayerPoseForEndSignals(window, lead: lead)
            let signals = walkingFrameSignals(keypoints: pose.keypoints, boxes: pose.boxes,
                                              homography: geometry.referenceHomography, fps: fps,
                                              onCourt: pose.onCourt)
            walkingProbability[lead] += walkingProbabilities(signals, windowStart: windowStart,
                                                             committing: featuresCommitted..<end)
            let signalsNearEnd = nearEndSignals(keypoints: pose.keypoints, boxes: pose.boxes, fps: fps,
                                                speed: signals.speed)
            for name in NearEndRules.SIGNAL_NAMES {
                nearEnd[lead][name]! += signalsNearEnd[name]![rows].map(toFloat32Precision)
            }
        }
        featuresCommitted = end
    }

    /// The walking probability for global samples `committing`, from frame
    /// signals of the window starting at `windowStart`. Decisions are taken
    /// every `decisionStride` samples counted from the start of the CLIP, and
    /// each sample takes the most recent decision -- as walking.predict does.
    private func walkingProbabilities(_ signals: WalkingFrameSignals, windowStart: Int,
                                      committing: Range<Int>) -> [Double] {
        let decisionStride = max(1, pythonRound(Double(walkingModel.post_stride) / Double(stride)))
        let firstDecision = committing.lowerBound / decisionStride * decisionStride
        let decisions = Array(Swift.stride(from: firstDecision, to: committing.upperBound, by: decisionStride))
        let (rows, _) = walkingWindowFeatures(signals, fps: fps, indices: decisions.map { $0 - windowStart })
        let probabilities = rows.map(walkingModel.walkingProbability)
        return committing.map { probabilities[($0 / decisionStride * decisionStride - firstDecision) / decisionStride] }
    }

    /// The smoothed live score for samples [liveCommitted, end), from the
    /// committed activity and end signals. Normalising it needs the whole
    /// clip, so that waits for Stop.
    private func commitLiveScore(through end: Int, finished: Bool) {
        guard end > liveCommitted else { return }
        let windowStart = max(0, liveCommitted - LiveLookahead.LIVE_SMOOTHING)
        let windowEnd = featuresCommitted
        let windowActivity = activity.map { Array($0[windowStart..<windowEnd]) }
        for lead in 0..<2 {
            let union = nonRallyUnion(EndSignals(
                fps: fps, walkingProbability: Array(walkingProbability[lead][windowStart..<windowEnd]),
                nearEnd: nearEnd[lead].mapValues { Array($0[windowStart..<windowEnd]) }))
            let smoothed = smoothedLiveScore(activity: windowActivity, union: union, fps: fps,
                                             smoothSeconds: reelConfig.live_smooth_s)
            smoothedLive[lead] += smoothed[(liveCommitted - windowStart)..<(end - windowStart)]
        }
        liveCommitted = end
    }
}
