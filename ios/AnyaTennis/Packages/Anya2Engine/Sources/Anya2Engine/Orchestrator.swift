// Orchestrator.swift  --  port of pipeline/anya2/orchestrator.py
//
// Turns the three detectors' events into the reel. THE REEL IS BUILT FROM
// SERVES, and ends only trim: serves are found far more reliably than ends,
// so every serve becomes a segment whether or not its end was seen.
//
// The recipe (numbers match buildReel below and the Python):
//    1 merge the near and far serve streams
//    2 drop serves struck while a point was already live
//    3 fold starts that are too close together to be two points
//    4 rule 1: a far serve's toss evidence moves its confidence
//    5 rule 2: a far serve surrounded by near serves is suspect
//    6 relabel sides so service games are at least 3 points long
//    7 note which half of the court each serve came from
//    8 give every start an end (detected, or estimated from this clip)
//    9 flag where deuce/ad alternation suggests a missed point
//   10 smooth: clamp, split overlaps, join near neighbours, drop flashes
//   11 add segments for long live play that no serve explained
//   12 smooth again
//
// Evidence and history: docs/anya2/NOTES.md, "orchestrator.py".
import Foundation

public struct ReelConfig: Codable {
    // 1, 3: merging the serve streams
    public var merge_window_s = 2.5      // near + far firing this close = one serve
    public var min_point_gap_s = 8.0     // two starts closer than this = one point
    // 6: service runs
    public var min_service_run = 3       // shortest service game (except first/last)
    public var enforce_runs = true
    public var drop_side_conflicts = false   // tested and rejected; keep false
    // 8: pairing serves with ends
    public var min_point_s = 2.5         // an end sooner than this is the serve itself
    public var max_point_s = 40.0        // an end later than this belongs to something else
    public var default_point_s = 9.0     // assumed point length with < 3 detected ends
    public var est_duration_pct = 85.0   // else: this percentile of the clip's own
    public var est_duration_pad_s = 1.5  //   detected point lengths, plus this
    public var next_start_guard_s = 4.0  // a segment's end stays this far before the next serve
    // 10: smoothing (pre/post roll is the biggest lever on what survives)
    public var pre_roll_s = 1.5          // footage kept before the serve (QUIRKS #5)
    public var post_roll_s = 1.0         // ...and after the end
    public var merge_gap_s = 6.0         // segments closer than this are joined
    public var min_segment_s = 4.0       // shorter segments are dropped (or padded)
    // 1, 8: extra thresholds on incoming events (0 = take all)
    public var near_threshold = 0.0
    public var far_threshold = 0.0
    public var end_threshold = 0.0
    public var use_near = true
    public var use_far = true
    public var use_end = true
    // 2: in-rally suppression
    public var live_gate_far = 0.8       // drop a far serve if the median live score
    public var live_gate_near = 0.0      //   over the previous 3 s exceeds this (0 = off)
    public var live_lookback_s = 3.0
    public var live_smooth_s: Double? = nil  // live-score smoothing; nil = PointEndRules.LIVE_SMOOTH_S
    // 11: recovering points with no serve
    public var recover_live_thr = 0.75   // live score counted as "clearly playing"
    public var recover_min_s = 3.0       // ...held at least this long
    // 4: rule 1, far toss evidence
    public var rule1_enabled = true
    public var toss_pose_lo = 0.30       // toss score at/below this = no support
    public var toss_pose_hi = 0.60       // ...at/above this = corroborating
    public var rule1_max_boost = 0.15    // most a strong toss adds to confidence
    public var rule1_max_penalty = 0.35  // most a missing toss subtracts
    // 5: rule 2, far serve among near serves
    public var rule2_enabled = true
    public var rule2_neighbours = 4      // look at this many surrounding starts
    public var rule2_min_near = 3        // ...penalise if at least this many are near
    public var rule2_penalty = 0.30

    public init() {}
}

public final class PointStart {
    public var t: Double                  // serve time, seconds
    public var side: String               // "near" | "far" | "both"
    public var p: Double                  // detector confidence
    public var track: Int?
    public var detectedSide = ""          // side before step 6 relabelled it
    public var sideConflict = false       // step 6 changed the side
    public var courtX: Double?            // metres from the centre line (step 7)
    public var tossPose: Double?          // far_serve's toss score
    public var tossCombined: Double?
    public var confidenceAdjustment = 0.0 // rules 1 and 2 add up here
    public var notes: [String] = []

    init(t: Double, side: String, p: Double, track: Int?, tossPose: Double? = nil) {
        self.t = t; self.side = side; self.p = p; self.track = track; self.tossPose = tossPose
    }

    /// Confidence after rules 1 and 2, clipped to [0, 1].
    public var adjustedP: Double { min(1, max(0, p + confidenceAdjustment)) }
}

public final class Segment: Codable {
    public var start: Double              // cut in (includes pre-roll)
    public var stop: Double               // cut out (includes post-roll)
    public var serveT: Double
    public var endT: Double
    public var side: String
    public var endSource: String          // "detected" | "estimated" | "recovered"
    public var p: Double
    public var notes: [String]

    init(start: Double, stop: Double, serveT: Double, endT: Double, side: String,
         endSource: String, p: Double = 0, notes: [String] = []) {
        self.start = start; self.stop = stop; self.serveT = serveT; self.endT = endT
        self.side = side; self.endSource = endSource; self.p = p; self.notes = notes
    }

    public var duration: Double { stop - start }
}

public struct Reel {
    public let starts: [PointStart]
    public let segments: [Segment]
    public var totalSeconds: Double { segments.reduce(0) { $0 + $1.duration } }
}

/// The recipe. `live` is the NORMALISED live score (normaliseLiveScore).
public func buildReel(near: [Event], far: [Event], ends: [Event], live: [Double]?, liveFPS: Double?,
                      tracks: PlayerTracks?, config cfg: ReelConfig = ReelConfig()) -> Reel {
    let duration = tracks?.durationSeconds
    var starts = mergeStarts(near: cfg.use_near ? near : [], far: cfg.use_far ? far : [], cfg)  // 1
    if let live, let liveFPS { starts = suppressInRally(starts, live: live, fps: liveFPS, cfg) } // 2
    starts = dropRapidRepeats(starts, cfg)                                               // 3
    // Rules 1-2 run before step 6: rule 2 reads the sides AS DETECTED.
    starts = applyTossRule(starts, cfg)                                                  // 4
    starts = applyNeighbourRule(starts, cfg)                                             // 5
    starts = enforceServiceRuns(starts, cfg)                                             // 6
    if let tracks { annotateServeCourt(starts, tracks: tracks) }                         // 7
    var segments = pairEnds(starts, ends: cfg.use_end ? ends : [], cfg, duration: duration)  // 8
    flagMissingPoints(starts, segments)                                                  // 9
    segments = smooth(segments, cfg, duration: duration)                                 // 10
    segments = recoverMissed(segments, live: live, fps: liveFPS, cfg, duration: duration)   // 11
    segments = smooth(segments, cfg, duration: duration)                                 // 12
    return Reel(starts: starts, segments: segments)
}

// MARK: - Steps 1-7: one timeline of point starts

/// 1. Union of the two serve streams, sorted. Serves within merge_window_s of
/// the previous kept one fold into it (side becomes "both"; the EARLIER time
/// is kept, because the two detectors anchor on different moments).
func mergeStarts(near: [Event], far: [Event], _ cfg: ReelConfig) -> [PointStart] {
    let candidates = stableSorted(
        near.filter { $0.p >= cfg.near_threshold }.map { PointStart(t: $0.t, side: "near", p: $0.p, track: $0.track) }
        + far.filter { $0.p >= cfg.far_threshold }.map {
            PointStart(t: $0.t, side: "far", p: $0.p, track: $0.track, tossPose: $0.detail["toss"])
        }) { $0.t < $1.t }
    var merged: [PointStart] = []
    for candidate in candidates {
        if let previous = merged.last, candidate.t - previous.t <= cfg.merge_window_s {
            if previous.side != candidate.side { previous.side = "both" }
            previous.p = max(previous.p, candidate.p)
            continue
        }
        merged.append(candidate)
    }
    return merged
}

/// 2. Drop a serve if the median live score over the preceding
/// live_lookback_s is above its side's gate (a point was already on).
func suppressInRally(_ starts: [PointStart], live: [Double], fps: Double, _ cfg: ReelConfig) -> [PointStart] {
    let n = live.count
    return starts.filter { start in
        let gate = start.side == "far" ? cfg.live_gate_far : cfg.live_gate_near
        guard gate > 0 else { return true }
        let windowStart = max(0, Int((start.t - cfg.live_lookback_s) * fps))
        let windowStop = max(windowStart + 1, min(n, Int(start.t * fps)))
        let window = windowStart < min(windowStop, n) ? Array(live[windowStart..<min(windowStop, n)]) : []
        return window.isEmpty || !(median(window) > gate)
    }
}

/// 3. Starts within min_point_gap_s of the previous kept one fold into it.
/// The FIRST is kept (a later echo would cut into the serve).
func dropRapidRepeats(_ starts: [PointStart], _ cfg: ReelConfig) -> [PointStart] {
    var kept: [PointStart] = []
    for start in starts {
        if let last = kept.last, start.t - last.t < cfg.min_point_gap_s {
            last.p = max(last.p, start.p)
            if last.side != start.side { last.side = "both" }
            continue
        }
        kept.append(start)
    }
    return kept
}

/// Python's f"{x:+.2f}" / f"{x:.2f}".
private func formatted(_ x: Double, digits: Int, signed: Bool = false) -> String {
    let text = String(format: "%.\(digits)f", pythonRound(x, digits: digits))
    return signed && !text.hasPrefix("-") ? "+" + text : text
}

/// 4. Rule 1: a far (or "both") serve's toss score moves its confidence:
/// toss <= toss_pose_lo -> -rule1_max_penalty ... toss >= toss_pose_hi ->
/// +rule1_max_boost, linear in between. Never removes a start.
func applyTossRule(_ starts: [PointStart], _ cfg: ReelConfig) -> [PointStart] {
    guard cfg.rule1_enabled else { return starts }
    for start in starts where start.side != "near" {
        guard let toss = start.tossPose else { continue }
        // -1 (no toss seen) .. +1 (clearly a toss)
        let support = 2 * clip((toss - cfg.toss_pose_lo) / max(cfg.toss_pose_hi - cfg.toss_pose_lo, 1e-6), 0, 1) - 1
        let adjustment = support >= 0 ? cfg.rule1_max_boost * support : cfg.rule1_max_penalty * support
        start.tossCombined = pythonRound(toss, digits: 4)
        start.confidenceAdjustment += adjustment
        start.notes.append("toss \(formatted(toss, digits: 2)) -> \(formatted(adjustment, digits: 2, signed: true))")
    }
    return starts
}

/// 5. Rule 2: penalise a far serve when most of its neighbouring starts (as
/// detected, before step 6) are near serves.
func applyNeighbourRule(_ starts: [PointStart], _ cfg: ReelConfig) -> [PointStart] {
    guard cfg.rule2_enabled && starts.count >= 3 else { return starts }
    let half = max(1, cfg.rule2_neighbours / 2)
    for (i, start) in starts.enumerated() where start.side == "far" {
        let neighbourSides = (max(0, i - half)..<min(starts.count, i + half + 1)).filter { $0 != i }.map { starts[$0].side }
        guard !neighbourSides.isEmpty else { continue }
        let nearCount = neighbourSides.filter { $0 == "near" }.count
        if nearCount >= cfg.rule2_min_near {
            start.confidenceAdjustment -= cfg.rule2_penalty
            start.notes.append("far serve among \(nearCount)/\(neighbourSides.count) near neighbours "
                               + "-> -\(formatted(cfg.rule2_penalty, digits: 2))")
        }
    }
    return starts
}

/// 6. Relabel sides so every service game is at least min_service_run long:
/// an exact dynamic programme (not greedy) over states (side, run length
/// capped at R, still in the first run). Relabelling costs 1 + confidence;
/// "both" is free either way. The first and last runs may be short. Only
/// labels change; no start is dropped. Ties keep the first-found state, as
/// Python's dict iteration does.
func enforceServiceRuns(_ startsIn: [PointStart], _ cfg: ReelConfig) -> [PointStart] {
    guard cfg.enforce_runs && startsIn.count >= 2 else { return startsIn }
    let sides = ["near", "far"]
    let runCap = max(1, cfg.min_service_run)
    struct State: Hashable { let side: Int; let run: Int; let inFirstRun: Bool }
    func relabelCost(_ start: PointStart, _ side: String) -> Double {
        start.side == "both" || start.side == side ? 0 : 1 + start.p
    }
    // Each column keeps its states in insertion order, like a Python dict.
    var table: [[(state: State, cost: Double, previous: State?)]] = [
        sides.indices.map { (State(side: $0, run: 1, inFirstRun: true), relabelCost(startsIn[0], sides[$0]), nil) }
    ]
    for k in 1..<startsIn.count {
        var column: [(state: State, cost: Double, previous: State?)] = []
        var position: [State: Int] = [:]
        func offer(_ state: State, _ cost: Double, _ previous: State) {
            if let at = position[state] {
                if cost < column[at].cost { column[at] = (state, cost, previous) }
            } else {
                position[state] = column.count
                column.append((state, cost, previous))
            }
        }
        for entry in table.last! {
            let s = entry.state
            // Same server continues.
            offer(State(side: s.side, run: min(s.run + 1, runCap), inFirstRun: s.inFirstRun),
                  entry.cost + relabelCost(startsIn[k], sides[s.side]), s)
            // Server changes -- allowed once the run is long enough (or in the first run).
            if s.run >= runCap || s.inFirstRun {
                let other = 1 - s.side
                offer(State(side: other, run: 1, inFirstRun: false),
                      entry.cost + relabelCost(startsIn[k], sides[other]), s)
            }
        }
        table.append(column)
    }
    var best = table.last![0]
    for entry in table.last! where entry.cost < best.cost { best = entry }
    var state: State? = best.state
    var labels = [String](repeating: "", count: startsIn.count)
    for k in stride(from: startsIn.count - 1, through: 0, by: -1) {
        labels[k] = sides[state!.side]
        state = table[k].first { $0.state == state! }!.previous
    }
    for (start, side) in zip(startsIn, labels) {
        start.detectedSide = start.side
        start.sideConflict = start.side != side && start.side != "both"
        start.side = side
    }
    return cfg.drop_side_conflicts ? startsIn.filter { !$0.sideConflict } : startsIn
}

/// 7. Record court_x (metres from the centre line) of the server over the
/// 3 s after each serve -- the deepest player on the serving side.
func annotateServeCourt(_ starts: [PointStart], tracks: PlayerTracks) {
    let n = tracks.samples
    for start in starts {
        let isNear = start.side == "near" || start.side == "both"
        let slots = isNear ? Slots.near : Slots.far
        let windowStart = max(0, Int(start.t * tracks.fps))
        let windowStop = min(n, Int((start.t + 3) * tracks.fps))
        guard windowStop > windowStart else { continue }
        var best: (deeper: Double, x: Double)? = nil
        for slot in slots {
            var xs: [Double] = [], ys: [Double] = []
            for i in windowStart..<windowStop {
                let x = Double(tracks.court[(i * Slots.count + slot) * 2])
                let y = Double(tracks.court[(i * Slots.count + slot) * 2 + 1])
                if x.isFinite && y.isFinite { xs.append(x); ys.append(y) }
            }
            guard !xs.isEmpty else { continue }
            let depth = median(ys)
            let deeperIsLarger = isNear ? -depth : depth
            if best == nil || deeperIsLarger > best!.deeper { best = (deeperIsLarger, median(xs)) }
        }
        if let best { start.courtX = best.x - Court.COURT_W / 2 }
    }
}

// MARK: - Steps 8-9: segments

/// 8. One segment per start, always. End = the first detected end in
/// [serve + min_point_s, serve + max_point_s] and before (next serve -
/// next_start_guard_s); if none, ESTIMATED as serve + this clip's typical
/// point length. Two passes: the first only learns that typical length.
func pairEnds(_ starts: [PointStart], ends: [Event], _ cfg: ReelConfig, duration: Double?) -> [Segment] {
    let endTimes = ends.filter { $0.p >= cfg.end_threshold }.map(\.t).sorted()
    let firstPass = pairOnce(starts, endTimes: endTimes, cfg, duration: duration, assumedPointSeconds: cfg.default_point_s)
    return pairOnce(starts, endTimes: endTimes, cfg, duration: duration,
                    assumedPointSeconds: estimatePointSeconds(firstPass, cfg))
}

/// Typical point length for this clip: est_duration_pct of the detected
/// durations + pad, or default_point_s with fewer than 3 detected.
func estimatePointSeconds(_ segments: [Segment], _ cfg: ReelConfig) -> Double {
    let detected = segments.filter { $0.endSource == "detected" && $0.endT > $0.serveT }.map { $0.endT - $0.serveT }
    guard detected.count >= 3 else { return cfg.default_point_s }
    return percentile(detected, cfg.est_duration_pct) + cfg.est_duration_pad_s
}

private func pairOnce(_ starts: [PointStart], endTimes: [Double], _ cfg: ReelConfig,
                      duration: Double?, assumedPointSeconds: Double) -> [Segment] {
    starts.enumerated().map { i, start in
        let nextServe = i + 1 < starts.count ? starts[i + 1].t : nil
        let earliest = start.t + cfg.min_point_s
        var latest = start.t + cfg.max_point_s
        if let nextServe { latest = min(latest, nextServe - cfg.next_start_guard_s) }
        var end: Double? = nil, source = ""
        if !endTimes.isEmpty && latest > earliest, let found = endTimes.first(where: { $0 >= earliest && $0 <= latest }) {
            end = found; source = "detected"
        }
        if end == nil {
            var estimate = start.t + assumedPointSeconds
            if let nextServe { estimate = min(estimate, nextServe - cfg.next_start_guard_s) }
            end = max(earliest, estimate); source = "estimated"
            if let duration { end = min(end!, duration - 0.1) }
        }
        return Segment(start: start.t - cfg.pre_roll_s, stop: end! + cfg.post_roll_s, serveT: start.t,
                       endT: end!, side: start.side, endSource: source, p: start.p)
    }
}

/// 9. Note (only) where two consecutive same-side serves came from the same
/// half of the court -- deuce/ad alternation suggests a point was missed.
func flagMissingPoints(_ starts: [PointStart], _ segments: [Segment]) {
    for i in 1..<max(starts.count, 1) {
        let previous = starts[i - 1], current = starts[i]
        guard previous.side == current.side, let a = previous.courtX, let b = current.courtX else { continue }
        if abs(a) < 0.35 || abs(b) < 0.35 { continue }   // too near the centre mark to call
        if a.sign == b.sign { segments[i].notes.append("same-court-as-previous: a point may be missing here") }
    }
}

// MARK: - Steps 10-12: making it watchable

/// 10/12. Three passes, in order:
///   a. clamp to the clip; split any overlap down the middle
///   b. join segments whose gap is <= merge_gap_s (a cut costs attention)
///   c. drop segments shorter than min_segment_s, unless they hold a detected
///      end -- those are padded to the minimum instead
func smooth(_ segmentsIn: [Segment], _ cfg: ReelConfig, duration: Double?) -> [Segment] {
    guard !segmentsIn.isEmpty else { return [] }
    let segments = stableSorted(segmentsIn) { $0.start < $1.start }
    for s in segments {
        s.start = max(0, s.start)
        if let duration { s.stop = min(s.stop, duration) }
        if s.stop <= s.start { s.stop = s.start + cfg.min_segment_s }
    }
    for (a, b) in zip(segments, segments.dropFirst()) where b.start < a.stop {
        let middle = 0.5 * (a.stop + b.start)
        a.stop = middle; b.start = middle
    }
    var joined = [segments[0]]
    for s in segments.dropFirst() {
        let previous = joined.last!
        if s.start - previous.stop <= cfg.merge_gap_s {
            previous.stop = max(previous.stop, s.stop)
            previous.endT = s.endT
            previous.notes.append("joined with the point at \(formatted(s.serveT, digits: 0))s")
            if previous.side != s.side { previous.side = "both" }
            continue
        }
        joined.append(s)
    }
    return joined.filter { s in
        guard s.duration < cfg.min_segment_s else { return true }
        if s.endSource == "detected" { s.stop = s.start + cfg.min_segment_s; return true }
        return false
    }
}

/// 11. Add a segment for every run of live score >= recover_live_thr lasting
/// at least recover_min_s that no existing segment covers. These carry no
/// side (no serve was seen) and end_source "recovered".
func recoverMissed(_ segments: [Segment], live: [Double]?, fps: Double?, _ cfg: ReelConfig,
                   duration: Double?) -> [Segment] {
    guard let live, let fps, fps > 0 else { return segments }
    let n = live.count
    var covered = [Bool](repeating: false, count: n)
    for s in segments {
        let a = max(0, Int(s.start * fps)), b = min(n, Int(s.stop * fps))
        if b > a { for i in a..<b { covered[i] = true } }
    }
    let minimumSamples = max(1, Int(cfg.recover_min_s * fps))
    var added: [Segment] = []
    for run in trueRuns((0..<n).map { live[$0] >= cfg.recover_live_thr && !covered[$0] })
    where run.stop - run.start >= minimumSamples {
        let t0 = Double(run.start) / fps, t1 = Double(run.stop) / fps
        let segment = Segment(start: max(0, t0 - cfg.pre_roll_s), stop: t1 + cfg.post_roll_s, serveT: t0,
                              endT: t1, side: "unknown", endSource: "recovered",
                              notes: ["recovered from live play; no serve was detected"])
        if let duration { segment.stop = min(segment.stop, duration) }
        added.append(segment)
    }
    return stableSorted(segments + added) { $0.start < $1.start }
}
