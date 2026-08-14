package com.build2launch.balltracker.tracking

import kotlin.math.sqrt

/** One detection offered to the solver, in analysis space (960-wide). */
data class ViterbiDetection(val x: Double, val y: Double, val conf: Double)

/** The motion models the solver may use to explain a change in velocity. */
enum class ViterbiMotion(val raw: String) {
    FLIGHT("flight"),  // smooth constant-velocity
    BOUNCE("bounce"),  // court bounce: vy reverses, vx roughly survives
    STRIKE("strike"),  // racket impact: velocity may change arbitrarily
}

/** One solved frame of the trajectory. */
data class ViterbiSample(
    val t: Double,
    val pos: Pair<Double, Double>?,
    val state: TrackState,
    val speedPxS: Double,          // analysis px/s
    val motion: ViterbiMotion?,    // null on the first frame of a segment / when untracked
)

/**
 * Tuning for the offline solver. Everything is in analysis-space pixels and
 * seconds; the solver maximises total score. Port of ios_tracker ViterbiConfig.
 */
data class ViterbiConfig(
    var topK: Int = 8,
    var maxGapFrames: Int = 15,
    var maxSpeedPxS: Double = 2200.0,
    var frameReward: Double = 2.0,
    var confWeight: Double = 2.0,
    var gapPenalty: Double = 0.3,
    var accelWeight: Double = 0.0015,
    var bouncePenalty: Double = 0.6,
    var strikePenalty: Double = 1.5,
    var strikeAccelWeight: Double = 0.0005,
    var restitution: Double = 0.6,
    var minBounceVy: Double = 60.0,
    var minSegmentSpanPx: Double = 40.0,
    var minSegmentFrames: Int = 4,
    var minSegmentScore: Double = 6.0,
    var anchorConf: Double = 0.25,
    var minAnchors: Int = 3,
    var maxSegments: Int = 64,
)

/**
 * Offline, whole-clip ball tracker.
 *
 * The online `BallTrackManager` commits to one association per frame and can
 * never revisit it. This solver instead builds a trellis over *every* plausible
 * detection-to-detection link in the clip and picks the trajectory that best
 * explains the evidence, so a bad early link can be undone by later frames.
 *
 * States are detection *pairs* (edges), not detections: a pair carries a
 * velocity, which is what makes acceleration — and therefore bounce and strike
 * — a well-defined cost.
 *
 * Port of ios_tracker BallTracker/Tracking/ViterbiTracker.swift.
 */
class ViterbiBallTracker(private val cfg: ViterbiConfig = ViterbiConfig()) {

    private class Node(val frame: Int, val x: Double, val y: Double, val conf: Double)

    /** An edge is a hypothesis "the ball went from node a to node b". */
    private class Edge(val a: Int, val b: Int, val vx: Double, val vy: Double, val dt: Double)

    /** Solve the whole clip. `frames[i]` are the detections at `times[i]`. */
    fun solve(frames: List<List<ViterbiDetection>>, times: List<Double>): List<ViterbiSample> {
        if (frames.size != times.size || frames.isEmpty()) return emptyList()

        // Flatten to nodes, keeping only the strongest few per frame.
        val nodes = ArrayList<Node>()
        val nodesByFrame = Array(frames.size) { ArrayList<Int>() }
        for (f in frames.indices) {
            for (d in frames[f].sortedByDescending { it.conf }.take(cfg.topK)) {
                nodesByFrame[f].add(nodes.size)
                nodes.add(Node(f, d.x, d.y, d.conf))
            }
        }
        if (nodes.isEmpty()) return emptySamples(times)

        val edges = buildEdges(nodes, nodesByFrame, times)
        if (edges.isEmpty()) return emptySamples(times)

        // Index edges by the node they enter, so the DP can find a state's
        // predecessors in O(1).
        val edgesInto = Array(nodes.size) { ArrayList<Int>() }
        for (i in edges.indices) {
            edgesInto[edges[i].b].add(i)
        }
        // Process edges end-frame first so predecessors are always settled.
        val order = edges.indices.sortedBy { nodes[edges[it].b].frame }

        val used = BooleanArray(nodes.size)
        val segments = ArrayList<List<Int>>()   // each: node indices, ascending in frame

        for (pass in 0 until cfg.maxSegments) {
            val seg = bestPath(nodes, edges, order, edgesInto, used) ?: break
            if (seg.score < cfg.minSegmentScore) break
            val anchors = seg.path.count { nodes[it].conf >= cfg.anchorConf }
            if (seg.path.size >= cfg.minSegmentFrames &&
                anchors >= cfg.minAnchors &&
                span(seg.path, nodes) >= cfg.minSegmentSpanPx) {
                segments.add(seg.path)
            }
            // Retire this segment's detections whether or not it was kept, so
            // the next pass finds something new instead of a near-duplicate.
            for (n in seg.path) used[n] = true
        }

        return samples(segments, nodes, edges, times)
    }

    // MARK: - Trellis

    private fun buildEdges(nodes: List<Node>, nodesByFrame: Array<ArrayList<Int>>,
                           times: List<Double>): List<Edge> {
        val edges = ArrayList<Edge>()
        for (f in nodesByFrame.indices) {
            for (a in nodesByFrame[f]) {
                val maxF = minOf(f + cfg.maxGapFrames, nodesByFrame.size - 1)
                if (maxF <= f) continue
                for (f2 in (f + 1)..maxF) {
                    val dt = times[f2] - times[f]
                    if (dt <= 0) continue
                    for (b in nodesByFrame[f2]) {
                        val vx = (nodes[b].x - nodes[a].x) / dt
                        val vy = (nodes[b].y - nodes[a].y) / dt
                        if (sqrt(vx * vx + vy * vy) > cfg.maxSpeedPxS) continue
                        edges.add(Edge(a, b, vx, vy, dt))
                    }
                }
            }
        }
        return edges
    }

    private fun nodeScore(n: Node): Double = cfg.frameReward + cfg.confWeight * n.conf

    /** Cheapest explanation for velocity v1 becoming v2, and which model bought it. */
    private fun motionCost(v1x: Double, v1y: Double, v2x: Double, v2y: Double): Pair<Double, ViterbiMotion> {
        val dvx = v2x - v1x
        val dvy = v2y - v1y
        val dv = sqrt(dvx * dvx + dvy * dvy)

        var bestCost = cfg.accelWeight * dv
        var bestMotion = ViterbiMotion.FLIGHT

        // Bounce: only if the ball was descending (image y grows downward).
        if (v1y > cfg.minBounceVy) {
            val ex = v1x
            val ey = -cfg.restitution * v1y
            val rx = v2x - ex
            val ry = v2y - ey
            val residual = sqrt(rx * rx + ry * ry)
            val cost = cfg.bouncePenalty + cfg.accelWeight * residual
            if (cost < bestCost) { bestCost = cost; bestMotion = ViterbiMotion.BOUNCE }
        }

        // Strike: anything goes, for a price.
        val strike = cfg.strikePenalty + cfg.strikeAccelWeight * dv
        if (strike < bestCost) { bestCost = strike; bestMotion = ViterbiMotion.STRIKE }

        return Pair(bestCost, bestMotion)
    }

    private class PathResult(val path: List<Int>, val score: Double)

    /** Viterbi over edge-states, skipping retired nodes. */
    private fun bestPath(nodes: List<Node>, edges: List<Edge>, order: List<Int>,
                         edgesInto: Array<ArrayList<Int>>, used: BooleanArray): PathResult? {
        val n = edges.size
        val best = DoubleArray(n) { Double.NEGATIVE_INFINITY }
        val back = IntArray(n) { -1 }   // predecessor edge, -1 = segment start

        for (ei in order) {
            val e = edges[ei]
            if (used[e.a] || used[e.b]) continue
            val gap = nodes[e.b].frame - nodes[e.a].frame - 1
            val base = nodeScore(nodes[e.b]) - cfg.gapPenalty * gap.toDouble()

            // Option 1: start a fresh segment on this edge.
            var top = nodeScore(nodes[e.a]) + base
            var from = -1

            // Option 2: extend a path that arrives at e.a.
            for (pi in edgesInto[e.a]) {
                if (best[pi] == Double.NEGATIVE_INFINITY) continue
                val p = edges[pi]
                if (used[p.a]) continue
                val (cost, _) = motionCost(p.vx, p.vy, e.vx, e.vy)
                val s = best[pi] + base - cost
                if (s > top) { top = s; from = pi }
            }
            best[ei] = top
            back[ei] = from
        }

        var endEdge = -1
        var endScore = Double.NEGATIVE_INFINITY
        for (i in 0 until n) {
            if (best[i] > endScore) { endScore = best[i]; endEdge = i }
        }
        if (endEdge < 0 || endScore == Double.NEGATIVE_INFINITY) return null

        // Walk back to recover the node chain.
        val chain = ArrayList<Int>()
        var curEdge = endEdge
        while (true) {
            chain.add(edges[curEdge].b)
            val prev = back[curEdge]
            if (prev < 0) { chain.add(edges[curEdge].a); break }
            curEdge = prev
        }
        chain.reverse()
        return PathResult(chain, best[endEdge])
    }

    // MARK: - Output

    private fun span(path: List<Int>, nodes: List<Node>): Double {
        if (path.isEmpty()) return 0.0
        val f = nodes[path.first()]
        var maxD = 0.0
        for (i in path) {
            val dx = nodes[i].x - f.x
            val dy = nodes[i].y - f.y
            maxD = maxOf(maxD, sqrt(dx * dx + dy * dy))
        }
        return maxD
    }

    private fun emptySamples(times: List<Double>): List<ViterbiSample> =
        times.map { ViterbiSample(it, null, TrackState.NONE, 0.0, null) }

    /** Rasterise the solved segments back to one sample per frame. */
    private fun samples(segments: List<List<Int>>, nodes: List<Node>, edges: List<Edge>,
                        times: List<Double>): List<ViterbiSample> {
        val pos = arrayOfNulls<Pair<Double, Double>>(times.size)
        val state = Array(times.size) { TrackState.NONE }
        val motion = arrayOfNulls<ViterbiMotion>(times.size)
        // Which segment owns each frame, so speed is never differenced across a
        // boundary — two unrelated segments meeting would read as a teleport.
        val segId = IntArray(times.size) { -1 }

        for (sid in segments.indices) {
            val path = segments[sid]
            for (k in path.indices) {
                val nd = nodes[path[k]]
                pos[nd.frame] = Pair(nd.x, nd.y)
                state[nd.frame] = TrackState.MOVING
                segId[nd.frame] = sid

                if (k == 0) continue
                val prev = nodes[path[k - 1]]
                if (k >= 2) {
                    val p2 = nodes[path[k - 2]]
                    val dt1 = times[prev.frame] - times[p2.frame]
                    val dt2 = times[nd.frame] - times[prev.frame]
                    if (dt1 > 0 && dt2 > 0) {
                        val v1x = (prev.x - p2.x) / dt1
                        val v1y = (prev.y - p2.y) / dt1
                        val v2x = (nd.x - prev.x) / dt2
                        val v2y = (nd.y - prev.y) / dt2
                        motion[nd.frame] = motionCost(v1x, v1y, v2x, v2y).second
                    }
                }
                // Fill an occlusion gap.
                val gap = nd.frame - prev.frame
                if (gap > 1) {
                    for (g in 1 until gap) {
                        val f = prev.frame + g
                        val u = g.toDouble() / gap.toDouble()
                        pos[f] = Pair(prev.x + (nd.x - prev.x) * u, prev.y + (nd.y - prev.y) * u)
                        state[f] = TrackState.COASTING
                        segId[f] = sid
                    }
                }
            }
        }

        // Speed from the rasterised track, central difference where possible.
        val out = ArrayList<ViterbiSample>(times.size)
        for (f in times.indices) {
            var speed = 0.0
            val p = pos[f]
            if (p != null) {
                val sid = segId[f]
                val prev = if (f > 0 && segId[f - 1] == sid) pos[f - 1] else null
                val next = if (f + 1 < times.size && segId[f + 1] == sid) pos[f + 1] else null
                if (prev != null && next != null && times[f + 1] - times[f - 1] > 0) {
                    val dt = times[f + 1] - times[f - 1]
                    val ax = (next.first - prev.first) / dt
                    val ay = (next.second - prev.second) / dt
                    speed = sqrt(ax * ax + ay * ay)
                } else if (prev != null && times[f] - times[f - 1] > 0) {
                    val dt = times[f] - times[f - 1]
                    val ax = (p.first - prev.first) / dt
                    val ay = (p.second - prev.second) / dt
                    speed = sqrt(ax * ax + ay * ay)
                }
            }
            out.add(ViterbiSample(times[f], pos[f], state[f], speed, motion[f]))
        }
        return out
    }
}
