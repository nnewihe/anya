import Foundation

/// Port of pipeline/rally_detector.py — trace-driven rally segmentation.
///
/// Segment rules
/// -------------
///   * A segment starts when the ball trace becomes active (hasMovingTrace)
///     and the ball is not "carried" by a walking player.
///   * A segment ends when the trace goes inactive (with a 1 s end-pad).
///   * Raw segments whose gap is < 4 s are merged into one segment.
///   * A 1.5 s pre-roll is prepended to each final (post-merge) segment start.
///
/// Post-processing (applied to merged segments)
/// --------------------------------------------
///   * Serving-pattern HMM — serve sides are sticky (same server for a whole
///     game). A Viterbi-decoded HMM over the segment sequence finds segments
///     whose observed origin disagrees with the inferred serving side; weak
///     disagreeing segments are dropped as spurious, strong ones are kept and
///     relabelled to the decoded side.

enum RallyConstants {
    static let gapThresholdSec = 4.0
    static let preRollSec      = 1.5
    static let endPadSec       = 1.0

    // ── Player-carry (velocity-coupling) suppression ──────────────────────
    // When the near player walks while holding the ball, the ball's pixel
    // velocity is coupled to the body's — same direction, same magnitude.
    // When struck, it decouples. coupling ratio = |v_ball - v_player| / |v_ball|
    //   ≈ 0  → carried (suppress);  ≈ 1+ → struck (keep).
    static let couplingWindowSec       = 0.40
    static let couplingMinPlayerSpeed  = 25.0   // px/s; player must be walking
    static let couplingRatioMax        = 0.50   // carried below this ratio

    // ── Serving-pattern HMM ───────────────────────────────────────────────
    // Fitted from 15 labeled matches (203 stay-, 14 switch-transitions).
    static let hmmPStay    = 0.9355   // average game ~15.5 points
    static let hmmPCorrect = 0.85     // per-segment origin label accuracy

    // Strength gate — a real point has at least one hard racket strike (the
    // serve), which spikes the IMM's racket_prob (μ₁) for several frames;
    // a ball-return is gentle and short. Strong if EITHER condition holds.
    static let racketSpikeThresh = 0.25
    static let minRacketFrames   = 5
    static let minSegmentSec     = 2.5
}

/// Which side served the point that opened a segment.
enum RallyOrigin: String, Codable {
    case near, far
}

/// A final rally segment in source-video seconds.
struct RallySegment {
    let start: Double
    let end: Double
    let origin: RallyOrigin
}

/// A player bounding box in the tracker's analysis space (top-left origin).
/// `vx`/`vy` are the box centroid's smoothed velocity in px/s, populated once
/// the player detector has tracked the box across frames (nil until then).
struct PlayerBox: Codable {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double
    var vx: Double?
    var vy: Double?

    var center: (x: Double, y: Double) { ((x1 + x2) / 2.0, (y1 + y2) / 2.0) }

    /// Smoothed box speed in px/s, or nil if the velocity is not yet known.
    var speed: Double? {
        guard let vx, let vy else { return nil }
        return (vx * vx + vy * vy).squareRoot()
    }

    func contains(x: Double, y: Double) -> Bool {
        x >= x1 && x <= x2 && y >= y1 && y <= y2
    }
}

/// Player boxes for one frame: every detected player in `all`, plus the near/
/// far pair the rally origin/HMM keys off (derived from `all`).
struct PlayerBoxes: Codable {
    var all: [PlayerBox]
    var near: PlayerBox?
    var far: PlayerBox?

    static let none = PlayerBoxes(all: [], near: nil, far: nil)
}

/// Drops ball detections that are being carried by a player — inside a moving
/// player box and sharing its pixel velocity — before they reach the tracker.
///
/// This generalises the near-player carry test in `RallyAccumulator` to every
/// detected player and moves it to the detection feed, so a held/carried ball
/// never seeds or sustains a track. It is deliberately conservative: a
/// detection is dropped only when its velocity can be estimated AND it couples
/// (`ratio < ratioMax`) to a box that is genuinely moving. A struck ball
/// decouples (high ratio) and is kept; a detection whose velocity can't be
/// estimated is kept.
///
/// coupling ratio = |v_det − v_box| / |v_det|  (≈0 carried, ≈1+ struck).
final class CarrySuppressor {
    let minPlayerSpeed: Double   // px/s; box must actually be moving
    let ratioMax: Double         // carried when ratio below this
    let matchGatePx: Double      // NN radius for estimating a detection's velocity

    private var prevDets: [TrackerDetection] = []
    private var prevT: Double?
    /// Detections dropped on the most recent `filter` call (debug/tuning).
    private(set) var lastSuppressedCount = 0

    /// Pipeline coupling constants are quoted in 960-wide analysis space; scale
    /// the pixel-valued ones to whatever analysis width the tracker runs in.
    init(analysisWidth: Double) {
        let s = analysisWidth / 960.0
        self.minPlayerSpeed = RallyConstants.couplingMinPlayerSpeed * s
        self.ratioMax = RallyConstants.couplingRatioMax
        self.matchGatePx = 30.0 * s
    }

    func filter(_ dets: [TrackerDetection], players: PlayerBoxes,
                t: Double) -> [TrackerDetection] {
        defer { prevDets = dets; prevT = t }
        lastSuppressedCount = 0
        guard let pT = prevT, t > pT, !players.all.isEmpty else { return dets }
        let dt = t - pT
        var kept: [TrackerDetection] = []
        kept.reserveCapacity(dets.count)
        for d in dets {
            if isCarried(d, dt: dt, players: players) {
                lastSuppressedCount += 1
            } else {
                kept.append(d)
            }
        }
        return kept
    }

    private func isCarried(_ d: TrackerDetection, dt: Double,
                           players: PlayerBoxes) -> Bool {
        // Only a moving box that actually contains the detection can carry it.
        let boxes = players.all.filter { box in
            guard let sp = box.speed, sp >= minPlayerSpeed else { return false }
            return box.contains(x: d.x, y: d.y)
        }
        guard !boxes.isEmpty, let v = detVelocity(d, dt: dt) else { return false }
        let speed = (v.0 * v.0 + v.1 * v.1).squareRoot()
        guard speed > 1e-6 else { return false }
        for box in boxes {
            guard let bvx = box.vx, let bvy = box.vy else { continue }
            let ratio = ((v.0 - bvx) * (v.0 - bvx) + (v.1 - bvy) * (v.1 - bvy)).squareRoot() / speed
            if ratio < ratioMax { return true }
        }
        return false
    }

    /// Estimate a detection's velocity from the nearest prior-frame detection
    /// within `matchGatePx`; nil when nothing is close enough to match.
    private func detVelocity(_ d: TrackerDetection, dt: Double) -> (Double, Double)? {
        guard dt > 0 else { return nil }
        var bestD = matchGatePx
        var best: TrackerDetection?
        for p in prevDets {
            let dist = ((d.x - p.x) * (d.x - p.x) + (d.y - p.y) * (d.y - p.y)).squareRoot()
            if dist <= bestD { bestD = dist; best = p }
        }
        guard let best else { return nil }
        return ((d.x - best.x) / dt, (d.y - best.y) / dt)
    }
}

/// Sliding-window velocity estimator over a stream of (t, x, y) pixel samples.
///
/// Velocity is the displacement between the oldest and newest samples inside a
/// `windowSec` window divided by their time span — a cheap, jitter-tolerant
/// smoother. Samples older than the window are pruned on every add.
final class SmoothedVelocity {
    private let windowSec: Double
    private var pts: [(t: Double, x: Double, y: Double)] = []

    init(windowSec: Double) {
        self.windowSec = windowSec
    }

    func add(t: Double, x: Double, y: Double) {
        pts.append((t, x, y))
        let cutoff = t - windowSec
        while let first = pts.first, first.t < cutoff {
            pts.removeFirst()
        }
    }

    /// (vx, vy) in px/s over the window, or nil if span/samples insufficient.
    func velocity() -> (vx: Double, vy: Double)? {
        guard pts.count >= 2, let p0 = pts.first, let p1 = pts.last else { return nil }
        let dt = p1.t - p0.t
        guard dt > 0 else { return nil }
        return ((p1.x - p0.x) / dt, (p1.y - p0.y) / dt)
    }
}

/// Streaming rally-segment accumulator plus the post-processing pipeline.
///
/// Feed one `observe` per processed frame (ball tracker status + player boxes),
/// then call `finish` once to close any open segment and run merge → HMM →
/// pre-roll, yielding the final segments.
final class RallyAccumulator {
    private struct Strength {
        var racketFrames: Int
        var duration: Double
    }

    private typealias RawSegment = (start: Double, end: Double,
                                    origin: RallyOrigin, strength: Strength)

    // Smoothed velocity windows for the carry-coupling test. The ball window
    // is fed the IMM-smoothed position; the player window the near-player-box
    // centroid — both every frame, so they stay populated across the
    // carry → strike transition.
    private let ballVel   = SmoothedVelocity(windowSec: RallyConstants.couplingWindowSec)
    private let playerVel = SmoothedVelocity(windowSec: RallyConstants.couplingWindowSec)

    private var rawSegments: [RawSegment] = []
    private var segStart: Double?
    private var segOrigin: RallyOrigin = .near
    private var segRacketFrames = 0
    private var lastT = 0.0
    private var lastDetectionTime: Double?

    /// Whether the last observed frame's trace was suppressed as a carried ball
    /// (exposed for debug overlays / harness logging).
    private(set) var lastCarried = false
    private(set) var lastCouplingRatio: Double?

    func observe(status: TrackStatus, players: PlayerBoxes, t: Double,
                 lastDetectionTime: Double?) {
        lastT = t
        self.lastDetectionTime = lastDetectionTime

        if let pos = status.position {
            ballVel.add(t: t, x: pos.x, y: pos.y)
        }
        if let near = players.near {
            let c = near.center
            playerVel.add(t: t, x: c.x, y: c.y)
        }

        // Suppress balls whose velocity is coupled to a walking player
        // (carried, not struck).
        var carried = false
        var ratio: Double?
        if status.hasMovingTrace, status.position != nil {
            (carried, ratio) = Self.isCarried(vBall: ballVel.velocity(),
                                              vPlayer: playerVel.velocity())
        }
        lastCarried = carried
        lastCouplingRatio = ratio
        let traceActive = status.hasMovingTrace && !carried

        // Accumulate per-segment strength while a segment is open.
        if segStart != nil, traceActive,
           status.racketProb > RallyConstants.racketSpikeThresh {
            segRacketFrames += 1
        }

        if traceActive {
            if segStart == nil {
                segStart = t
                segOrigin = Self.originSide(ballXY: status.position,
                                            nearBox: players.near,
                                            farBox: players.far)
                segRacketFrames = 0
            }
        } else if segStart != nil {
            closeOpenSegment(fallbackEnd: t, videoDuration: .infinity)
        }
    }

    /// Close any open segment and run the post-processing pipeline:
    /// merge close gaps → HMM serving-pattern filter → pre-roll.
    func finish(videoDuration: Double) -> [RallySegment] {
        if segStart != nil {
            closeOpenSegment(fallbackEnd: lastT, videoDuration: videoDuration)
        }
        print("[RALLY] Raw segments: \(rawSegments.count)")

        let merged = Self.merge(rawSegments)
        print("[RALLY] After merging (gap < \(Int(RallyConstants.gapThresholdSec))s): "
              + "\(merged.count) segment(s)")

        let filtered = Self.hmmFilter(merged)
        print("[RALLY] After HMM filter: \(filtered.count) segment(s)")

        let final = filtered.map { seg in
            RallySegment(start: max(0.0, seg.start - RallyConstants.preRollSec),
                         end: min(seg.end, videoDuration),
                         origin: seg.origin)
        }
        let nFar = final.filter { $0.origin == .far }.count
        print("[RALLY] Segment origins: \(final.count - nFar) near, \(nFar) far")
        return final
    }

    private func closeOpenSegment(fallbackEnd: Double, videoDuration: Double) {
        guard let start = segStart else { return }
        let rawEnd = lastDetectionTime ?? fallbackEnd
        let paddedEnd = min(rawEnd + RallyConstants.endPadSec, videoDuration)
        rawSegments.append((start, paddedEnd, segOrigin,
                            Strength(racketFrames: segRacketFrames,
                                     duration: rawEnd - start)))
        segStart = nil
    }

    // MARK: Carry suppression

    /// ratio = |v_ball - v_player| / |v_ball|; nil when it cannot be computed.
    /// playerSpeed (px/s) is returned alongside so the caller can require the
    /// player to actually be moving before declaring a carried ball.
    static func couplingRatio(vBall: (vx: Double, vy: Double)?,
                              vPlayer: (vx: Double, vy: Double)?)
        -> (ratio: Double?, playerSpeed: Double) {
        guard let b = vBall, let p = vPlayer else { return (nil, 0.0) }
        let ballSpeed = (b.vx * b.vx + b.vy * b.vy).squareRoot()
        let playerSpeed = (p.vx * p.vx + p.vy * p.vy).squareRoot()
        guard ballSpeed >= 1e-6 else { return (nil, playerSpeed) }
        let dx = b.vx - p.vx
        let dy = b.vy - p.vy
        return ((dx * dx + dy * dy).squareRoot() / ballSpeed, playerSpeed)
    }

    /// Carried when the player is genuinely moving and the ball's velocity
    /// closely matches it. Ratio is surfaced for debug output.
    static func isCarried(vBall: (vx: Double, vy: Double)?,
                          vPlayer: (vx: Double, vy: Double)?)
        -> (carried: Bool, ratio: Double?) {
        let (ratio, playerSpeed) = couplingRatio(vBall: vBall, vPlayer: vPlayer)
        guard let ratio else { return (false, nil) }
        let carried = ratio < RallyConstants.couplingRatioMax
            && playerSpeed >= RallyConstants.couplingMinPlayerSpeed
        return (carried, ratio)
    }

    // MARK: Origin classification

    /// Classify a serve's origin as near or far by which player box the ball is
    /// closest to at the moment the segment opens. Falls back to near when
    /// boxes are unavailable (conservative: ambiguous cases stay near-owned).
    static func originSide(ballXY: (x: Double, y: Double)?,
                           nearBox: PlayerBox?, farBox: PlayerBox?) -> RallyOrigin {
        guard let ball = ballXY else { return .near }
        guard let farC = farBox?.center else { return .near }
        guard let nearC = nearBox?.center else { return .far }
        let dNear = hypot(ball.x - nearC.x, ball.y - nearC.y)
        let dFar  = hypot(ball.x - farC.x,  ball.y - farC.y)
        return dFar < dNear ? .far : .near
    }

    // MARK: Merge

    /// Merge adjacent segments whose gap < gapThresholdSec. The merged segment
    /// keeps the origin of its FIRST sub-segment (the serve that opened the
    /// run) and accumulates strength.
    private static func merge(_ segments: [RawSegment]) -> [RawSegment] {
        guard var current = segments.first else { return [] }
        var merged: [RawSegment] = []
        for seg in segments.dropFirst() {
            if seg.start - current.end < RallyConstants.gapThresholdSec {
                current.end = seg.end
                current.strength.racketFrames += seg.strength.racketFrames
                current.strength.duration     += seg.strength.duration
            } else {
                merged.append(current)
                current = seg
            }
        }
        merged.append(current)
        return merged
    }

    // MARK: Serving-pattern HMM

    private static func isStrong(_ s: Strength) -> Bool {
        s.racketFrames >= RallyConstants.minRacketFrames
            || s.duration >= RallyConstants.minSegmentSec
    }

    /// Viterbi decoding of the most likely serving-side state sequence.
    /// States/observations: near=0, far=1. Symmetric sticky transitions,
    /// P(obs=state) = pCorrect emissions, uniform initial.
    static func viterbiDecode(_ obsSides: [RallyOrigin],
                              pStay: Double = RallyConstants.hmmPStay,
                              pCorrect: Double = RallyConstants.hmmPCorrect)
        -> [RallyOrigin] {
        let n = obsSides.count
        if n == 0 { return [] }
        if n == 1 { return obsSides }   // single segment: no context, trust as-is

        let sides: [RallyOrigin] = [.near, .far]
        let logTrans = [[log(pStay), log(1.0 - pStay)],
                        [log(1.0 - pStay), log(pStay)]]
        let logEmit  = [[log(pCorrect), log(1.0 - pCorrect)],
                        [log(1.0 - pCorrect), log(pCorrect)]]
        let obs = obsSides.map { $0 == .near ? 0 : 1 }

        // Forward pass — delta[s] = log prob of best path ending in state s.
        var delta = [log(0.5) + logEmit[0][obs[0]],
                     log(0.5) + logEmit[1][obs[0]]]
        var psi = [[Int]](repeating: [0, 0], count: n)

        for t in 1..<n {
            var next = [0.0, 0.0]
            for s in 0..<2 {
                let scores = [delta[0] + logTrans[0][s], delta[1] + logTrans[1][s]]
                let best = scores[1] > scores[0] ? 1 : 0
                next[s] = scores[best] + logEmit[s][obs[t]]
                psi[t][s] = best
            }
            delta = next
        }

        // Backtrack.
        var path = [Int](repeating: 0, count: n)
        path[n - 1] = delta[1] > delta[0] ? 1 : 0
        for t in stride(from: n - 2, through: 0, by: -1) {
            path[t] = psi[t + 1][path[t + 1]]
        }
        return path.map { sides[$0] }
    }

    /// For each segment whose observed origin disagrees with the decoded
    /// serving side: weak → drop (spurious), strong → keep but relabel.
    /// Agreeing segments pass through unchanged.
    private static func hmmFilter(_ segments: [RawSegment])
        -> [(start: Double, end: Double, origin: RallyOrigin)] {
        guard !segments.isEmpty else { return [] }

        let decoded = viterbiDecode(segments.map(\.origin))
        var kept: [(start: Double, end: Double, origin: RallyOrigin)] = []
        var nDropped = 0
        var nRelabeled = 0

        for (seg, decodedSide) in zip(segments, decoded) {
            if seg.origin == decodedSide {
                kept.append((seg.start, seg.end, seg.origin))
            } else if isStrong(seg.strength) {
                kept.append((seg.start, seg.end, decodedSide))
                nRelabeled += 1
                print(String(format: "[HMM] Relabelled  %.2fs–%.2fs %@→%@  "
                             + "(racket_frames=%d, dur=%.1fs)",
                             seg.start, seg.end, seg.origin.rawValue,
                             decodedSide.rawValue, seg.strength.racketFrames,
                             seg.strength.duration))
            } else {
                nDropped += 1
                print(String(format: "[HMM] Dropped     %.2fs–%.2fs origin=%@ "
                             + "(decoded=%@)  (racket_frames=%d, dur=%.1fs) — weak/spurious",
                             seg.start, seg.end, seg.origin.rawValue,
                             decodedSide.rawValue, seg.strength.racketFrames,
                             seg.strength.duration))
            }
        }

        print("[HMM] Filter result: kept \(kept.count), "
              + "dropped \(nDropped), relabelled \(nRelabeled)")
        return kept
    }
}
