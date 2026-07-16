import CoreGraphics
import Foundation

/// One detection offered to the solver, in analysis space (960-wide).
struct ViterbiDetection {
    let x: Double
    let y: Double
    let conf: Double
}

/// One solved frame of the trajectory.
struct ViterbiSample {
    let t: Double
    let pos: (x: Double, y: Double)?
    let state: TrackState
    let speedPxS: Double        // analysis px/s
    /// Which motion model explains the step into this frame. Nil on the first
    /// frame of a segment, or when the ball isn't tracked.
    let motion: ViterbiMotion?
}

/// The motion models the solver may use to explain a change in velocity. These
/// are the cost-space counterpart of the online tracker's IMM: `BallTrackManager`
/// runs smooth-CV / racket / bounce filters and reports racketProb & bounceProb,
/// but nothing consumed them. Here the same three hypotheses are priced, and
/// the cheapest one that explains each step is what the trellis picks.
enum ViterbiMotion: String {
    case flight   // smooth constant-velocity
    case bounce   // court bounce: vy reverses, vx roughly survives
    case strike   // racket impact: velocity may change arbitrarily
}

/// Tuning for the offline solver. Everything is in analysis-space pixels and
/// seconds; the solver maximises total score.
struct ViterbiConfig {
    /// Detections kept per frame, best confidence first. Transitions are
    /// O(K²) per node, so this is the main cost knob.
    var topK = 8
    /// Frames a trajectory may skip (occlusion, a missed detection).
    ///
    /// Measured on real footage: ~48% of frames yield no ball detection at all,
    /// and runs of empty frames go p50=3, p90=14, max=50. Bridging 5 frames
    /// only covers 72% of gaps; 15 covers 94%, which is why this is not the
    /// "obvious" small number. Cost grows ~linearly with it.
    var maxGapFrames = 15
    /// Hard gate: nothing faster than this is a ball.
    var maxSpeedPxS: Double = 2200

    /// Paid per tracked frame; makes a long coherent path beat a short one.
    /// Must exceed the typical flight transition cost or nothing links up.
    var frameReward: Double = 2.0
    /// Weight on detection confidence in a node's score.
    var confWeight: Double = 2.0
    /// Charged per skipped frame. Low enough that bridging a real occlusion to
    /// reach more ball beats giving up, given gaps this common.
    var gapPenalty: Double = 0.3

    /// Score charged per (px/s) of unexplained velocity change in flight.
    /// Detection jitter of a few px at 30 fps is already ~130 px/s of apparent
    /// velocity noise, so this has to stay small or honest flight looks
    /// expensive and paths fragment.
    var accelWeight: Double = 0.0015
    /// Flat charge for invoking a bounce.
    var bouncePenalty: Double = 0.6
    /// Flat charge for invoking a racket strike. Higher than a bounce: strikes
    /// are rarer, and this stops the solver explaining every bad link as one.
    var strikePenalty: Double = 1.5
    /// Residual weight once a strike is paid for — deliberately slack, since a
    /// strike may redirect the ball arbitrarily.
    var strikeAccelWeight: Double = 0.0005
    /// Vertical restitution of a hard-court bounce.
    var restitution: Double = 0.6
    /// A bounce needs at least this much downward speed going in.
    var minBounceVy: Double = 60

    /// A segment must displace at least this far to count as a ball, in
    /// analysis px. Mirrors the online tracker's moveThreshPx and is what stops
    /// a stationary blob being "tracked".
    var minSegmentSpanPx: Double = 40
    /// Frames a segment needs before it is believable.
    var minSegmentFrames = 4
    /// A segment must beat this total score to be kept.
    var minSegmentScore: Double = 6.0

    /// A kept segment must contain at least `minAnchors` detections at or above
    /// `anchorConf`. This is what makes the low solver threshold survivable: at
    /// conf 0.03 most candidates are noise, and without an anchor requirement
    /// the trellis happily stitches weak clutter into a smooth, entirely
    /// fictional trajectory across empty court. A real rally always contains
    /// several confident looks at the ball; a noise chain contains none.
    var anchorConf: Double = 0.25
    var minAnchors = 3
    /// Cap on extracted segments (rallies) per clip.
    var maxSegments = 64

    /// Overlay any `VITERBI_*` environment variables onto the defaults, so the
    /// tuning sweep can explore weights without a recompile per trial. Names
    /// match the fields, upper-snake-cased: VITERBI_ACCEL_WEIGHT, etc.
    static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment)
        -> ViterbiConfig {
        var c = ViterbiConfig()
        func dbl(_ k: String, _ set: (Double) -> Void) {
            if let v = env["VITERBI_" + k], let d = Double(v) { set(d) }
        }
        func int(_ k: String, _ set: (Int) -> Void) {
            if let v = env["VITERBI_" + k], let i = Int(v) { set(i) }
        }
        int("TOP_K") { c.topK = $0 }
        int("MAX_GAP_FRAMES") { c.maxGapFrames = $0 }
        dbl("MAX_SPEED_PXS") { c.maxSpeedPxS = $0 }
        dbl("FRAME_REWARD") { c.frameReward = $0 }
        dbl("CONF_WEIGHT") { c.confWeight = $0 }
        dbl("GAP_PENALTY") { c.gapPenalty = $0 }
        dbl("ACCEL_WEIGHT") { c.accelWeight = $0 }
        dbl("BOUNCE_PENALTY") { c.bouncePenalty = $0 }
        dbl("STRIKE_PENALTY") { c.strikePenalty = $0 }
        dbl("STRIKE_ACCEL_WEIGHT") { c.strikeAccelWeight = $0 }
        dbl("RESTITUTION") { c.restitution = $0 }
        dbl("MIN_BOUNCE_VY") { c.minBounceVy = $0 }
        dbl("MIN_SEGMENT_SPAN_PX") { c.minSegmentSpanPx = $0 }
        int("MIN_SEGMENT_FRAMES") { c.minSegmentFrames = $0 }
        dbl("MIN_SEGMENT_SCORE") { c.minSegmentScore = $0 }
        dbl("ANCHOR_CONF") { c.anchorConf = $0 }
        int("MIN_ANCHORS") { c.minAnchors = $0 }
        int("MAX_SEGMENTS") { c.maxSegments = $0 }
        return c
    }
}

/// Offline, whole-clip ball tracker.
///
/// The online `BallTrackManager` commits to one association per frame and can
/// never revisit it. This solver instead builds a trellis over *every* plausible
/// detection-to-detection link in the clip and picks the trajectory that best
/// explains the evidence, so a bad early link can be undone by later frames.
///
/// States are detection *pairs*, not detections: a pair carries a velocity,
/// which is what makes acceleration — and therefore bounce and strike — a
/// well-defined cost. A first-order chain over positions could not tell a
/// bounce from a teleport.
///
/// Rallies are separated by dead time, so rather than one global path the
/// solver extracts segments greedily: best path, mask its detections, repeat
/// until nothing worthwhile is left.
final class ViterbiBallTracker {
    private let cfg: ViterbiConfig

    init(config: ViterbiConfig = ViterbiConfig()) {
        self.cfg = config
    }

    private struct Node {
        let frame: Int
        let x: Double
        let y: Double
        let conf: Double
    }

    /// An edge is a hypothesis "the ball went from node `a` to node `b`", and
    /// carries the implied velocity.
    private struct Edge {
        let a: Int          // node index
        let b: Int
        let vx: Double
        let vy: Double
        let dt: Double
    }

    /// Solve the whole clip. `frames[i]` are the detections at `times[i]`.
    func solve(frames: [[ViterbiDetection]], times: [Double]) -> [ViterbiSample] {
        guard frames.count == times.count, !frames.isEmpty else { return [] }

        // Flatten to nodes, keeping only the strongest few per frame.
        var nodes: [Node] = []
        var nodesByFrame: [[Int]] = Array(repeating: [], count: frames.count)
        for (f, dets) in frames.enumerated() {
            for d in dets.sorted(by: { $0.conf > $1.conf }).prefix(cfg.topK) {
                nodesByFrame[f].append(nodes.count)
                nodes.append(Node(frame: f, x: d.x, y: d.y, conf: d.conf))
            }
        }
        guard !nodes.isEmpty else { return emptySamples(times) }

        let edges = buildEdges(nodes: nodes, nodesByFrame: nodesByFrame, times: times)
        guard !edges.isEmpty else { return emptySamples(times) }

        // Index edges by the node they leave and the node they enter, so the
        // DP can find a state's predecessors in O(1).
        var edgesInto = [[Int]](repeating: [], count: nodes.count)
        var edgesOutOf = [[Int]](repeating: [], count: nodes.count)
        for (i, e) in edges.enumerated() {
            edgesInto[e.b].append(i)
            edgesOutOf[e.a].append(i)
        }
        // Process edges end-frame first so predecessors are always settled.
        let order = edges.indices.sorted { nodes[edges[$0].b].frame < nodes[edges[$1].b].frame }

        var used = [Bool](repeating: false, count: nodes.count)
        var segments: [[Int]] = []   // each: node indices, ascending in frame

        for _ in 0..<cfg.maxSegments {
            guard let seg = bestPath(nodes: nodes, edges: edges, order: order,
                                     edgesInto: edgesInto, used: used) else { break }
            if seg.score < cfg.minSegmentScore { break }
            let anchors = seg.path.filter { nodes[$0].conf >= cfg.anchorConf }.count
            if seg.path.count >= cfg.minSegmentFrames,
               anchors >= cfg.minAnchors,
               span(of: seg.path, nodes: nodes) >= cfg.minSegmentSpanPx {
                segments.append(seg.path)
            }
            // Retire this segment's detections whether or not it was kept, so
            // the next pass finds something new instead of a near-duplicate.
            for n in seg.path { used[n] = true }
            _ = edgesOutOf   // (kept for clarity of the index pair)
        }

        return samples(from: segments, nodes: nodes, edges: edges, times: times)
    }

    // MARK: - Trellis

    private func buildEdges(nodes: [Node], nodesByFrame: [[Int]], times: [Double]) -> [Edge] {
        var edges: [Edge] = []
        for (f, ids) in nodesByFrame.enumerated() {
            for a in ids {
                let maxF = min(f + cfg.maxGapFrames, nodesByFrame.count - 1)
                guard maxF > f else { continue }
                for f2 in (f + 1)...maxF {
                    let dt = times[f2] - times[f]
                    guard dt > 0 else { continue }
                    for b in nodesByFrame[f2] {
                        let vx = (nodes[b].x - nodes[a].x) / dt
                        let vy = (nodes[b].y - nodes[a].y) / dt
                        if (vx * vx + vy * vy).squareRoot() > cfg.maxSpeedPxS { continue }
                        edges.append(Edge(a: a, b: b, vx: vx, vy: vy, dt: dt))
                    }
                }
            }
        }
        return edges
    }

    private func nodeScore(_ n: Node) -> Double {
        cfg.frameReward + cfg.confWeight * n.conf
    }

    /// Cheapest explanation for velocity `v1` becoming `v2`, and which model
    /// bought it. This is where bounce and strike enter the objective: both are
    /// *allowed*, each at a price, so the trellis can route a trajectory through
    /// a genuine direction change without paying the (much larger) cost of
    /// pretending it was smooth flight.
    private func motionCost(v1: (x: Double, y: Double),
                            v2: (x: Double, y: Double)) -> (cost: Double, motion: ViterbiMotion) {
        let dvx = v2.x - v1.x, dvy = v2.y - v1.y
        let dv = (dvx * dvx + dvy * dvy).squareRoot()

        var best = (cost: cfg.accelWeight * dv, motion: ViterbiMotion.flight)

        // Bounce: only if the ball was descending (image y grows downward).
        // Expect vy to reverse and shed energy, vx to survive.
        if v1.y > cfg.minBounceVy {
            let ex = v1.x
            let ey = -cfg.restitution * v1.y
            let rx = v2.x - ex, ry = v2.y - ey
            let residual = (rx * rx + ry * ry).squareRoot()
            let cost = cfg.bouncePenalty + cfg.accelWeight * residual
            if cost < best.cost { best = (cost, .bounce) }
        }

        // Strike: anything goes, for a price.
        let strike = cfg.strikePenalty + cfg.strikeAccelWeight * dv
        if strike < best.cost { best = (strike, .strike) }

        return best
    }

    private struct PathResult {
        let path: [Int]
        let score: Double
    }

    /// Viterbi over edge-states, skipping retired nodes. Returns the highest
    /// scoring trajectory currently available.
    private func bestPath(nodes: [Node], edges: [Edge], order: [Int],
                          edgesInto: [[Int]], used: [Bool]) -> PathResult? {
        let n = edges.count
        var best = [Double](repeating: -.infinity, count: n)
        var back = [Int](repeating: -1, count: n)   // predecessor edge, -1 = segment start

        for ei in order {
            let e = edges[ei]
            if used[e.a] || used[e.b] { continue }
            let gap = nodes[e.b].frame - nodes[e.a].frame - 1
            let base = nodeScore(nodes[e.b]) - cfg.gapPenalty * Double(gap)

            // Option 1: start a fresh segment on this edge.
            var top = nodeScore(nodes[e.a]) + base
            var from = -1

            // Option 2: extend a path that arrives at e.a.
            for pi in edgesInto[e.a] where best[pi] > -.infinity {
                let p = edges[pi]
                if used[p.a] { continue }
                let (cost, _) = motionCost(v1: (p.vx, p.vy), v2: (e.vx, e.vy))
                let s = best[pi] + base - cost
                if s > top { top = s; from = pi }
            }
            best[ei] = top
            back[ei] = from
        }

        guard let endEdge = best.indices.filter({ best[$0] > -.infinity })
            .max(by: { best[$0] < best[$1] }) else { return nil }
        guard best[endEdge] > -.infinity else { return nil }

        // Walk back to recover the node chain.
        var chain: [Int] = []
        var cur = endEdge
        while true {
            chain.append(edges[cur].b)
            let prev = back[cur]
            if prev < 0 { chain.append(edges[cur].a); break }
            cur = prev
        }
        return PathResult(path: chain.reversed(), score: best[endEdge])
    }

    // MARK: - Output

    private func span(of path: [Int], nodes: [Node]) -> Double {
        guard let f = path.first.map({ nodes[$0] }) else { return 0 }
        var maxD = 0.0
        for i in path {
            let dx = nodes[i].x - f.x, dy = nodes[i].y - f.y
            maxD = max(maxD, (dx * dx + dy * dy).squareRoot())
        }
        return maxD
    }

    private func emptySamples(_ times: [Double]) -> [ViterbiSample] {
        times.map { ViterbiSample(t: $0, pos: nil, state: .none, speedPxS: 0, motion: nil) }
    }

    /// Rasterise the solved segments back to one sample per frame, filling
    /// skipped frames by interpolation and marking them `.coasting` — the frame
    /// had no detection of its own, matching what the online tracker reports.
    private func samples(from segments: [[Int]], nodes: [Node], edges: [Edge],
                         times: [Double]) -> [ViterbiSample] {
        var pos = [(x: Double, y: Double)?](repeating: nil, count: times.count)
        var state = [TrackState](repeating: .none, count: times.count)
        var motion = [ViterbiMotion?](repeating: nil, count: times.count)
        // Which segment owns each frame, so speed is never differenced across
        // a boundary — two unrelated segments meeting would read as a teleport.
        var segId = [Int](repeating: -1, count: times.count)

        for (sid, path) in segments.enumerated() {
            for (k, id) in path.enumerated() {
                let nd = nodes[id]
                pos[nd.frame] = (nd.x, nd.y)
                state[nd.frame] = .moving
                segId[nd.frame] = sid

                guard k > 0 else { continue }
                let prev = nodes[path[k - 1]]
                if k >= 2 {
                    let p2 = nodes[path[k - 2]]
                    let dt1 = times[prev.frame] - times[p2.frame]
                    let dt2 = times[nd.frame] - times[prev.frame]
                    if dt1 > 0, dt2 > 0 {
                        let v1 = ((prev.x - p2.x) / dt1, (prev.y - p2.y) / dt1)
                        let v2 = ((nd.x - prev.x) / dt2, (nd.y - prev.y) / dt2)
                        motion[nd.frame] = motionCost(v1: v1, v2: v2).motion
                    }
                }
                // Fill an occlusion gap.
                let gap = nd.frame - prev.frame
                if gap > 1 {
                    for g in 1..<gap {
                        let f = prev.frame + g
                        let u = Double(g) / Double(gap)
                        pos[f] = (prev.x + (nd.x - prev.x) * u, prev.y + (nd.y - prev.y) * u)
                        state[f] = .coasting
                        segId[f] = sid
                    }
                }
            }
        }

        // Speed from the rasterised track, central difference where possible.
        var out: [ViterbiSample] = []
        for f in times.indices {
            var speed = 0.0
            if let p = pos[f] {
                let sid = segId[f]
                let prev = (f > 0 && segId[f - 1] == sid) ? pos[f - 1] : nil
                let next = (f + 1 < times.count && segId[f + 1] == sid) ? pos[f + 1] : nil
                if let a = prev, let b = next, times[f + 1] - times[f - 1] > 0 {
                    let dt = times[f + 1] - times[f - 1]
                    speed = (((b.x - a.x) / dt) * ((b.x - a.x) / dt)
                             + ((b.y - a.y) / dt) * ((b.y - a.y) / dt)).squareRoot()
                } else if let a = prev, times[f] - times[f - 1] > 0 {
                    let dt = times[f] - times[f - 1]
                    speed = (((p.x - a.x) / dt) * ((p.x - a.x) / dt)
                             + ((p.y - a.y) / dt) * ((p.y - a.y) / dt)).squareRoot()
                }
            }
            out.append(ViterbiSample(t: times[f], pos: pos[f], state: state[f],
                                     speedPxS: speed, motion: motion[f]))
        }
        return out
    }
}
