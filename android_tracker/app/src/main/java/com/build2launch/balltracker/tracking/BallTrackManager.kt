package com.build2launch.balltracker.tracking

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** A pixel-centre detection plus its YOLO confidence. */
data class TrackerDetection(val x: Double, val y: Double, val conf: Double)

typealias PerspectiveScale = (Double) -> Double

private val noPerspective: PerspectiveScale = { 1.0 }

/**
 * Cheap perspective model needing only the analysis-frame height. Multiplier
 * in (farFloor, 1.0]: ~1.0 near the bottom, shrinking toward farFloor at top.
 */
fun makeImageRowPerspective(frameHeight: Double, farFloor: Double = 0.35): PerspectiveScale {
    val h = max(1.0, frameHeight)
    return { y -> max(farFloor, min(1.0, y / h)) }
}

enum class TrackState(val raw: String) {
    NONE("none"), FADING("fading"), STOPPED("stopped"),
    MOVING("moving"), COASTING("coasting"), LOST("lost")
}

/**
 * Per-frame answer handed back to the caller.
 * Port of mobile/lib/engine/ball_tracker.dart TrackStatus.
 */
data class TrackStatus(
    val hasMovingTrace: Boolean,
    val state: TrackState,
    val position: Pair<Double, Double>?,
    val speedPxS: Double,
    val timeSinceDetection: Double,
    val coasting: Boolean,
    val ballCount: Int,
    val maneuverProb: Double,
    val racketProb: Double,
    val bounceProb: Double,
    val trace: List<Pair<Double, Double>>,
)

private class ConfirmedTrack(
    val fps: Double,
    x: Double,
    y: Double,
    vx: Double,
    vy: Double,
    t: Double,
    val motionWindowS: Double,
    val corroborationWindowS: Double,
    qSmooth: Double = 5.0,
    qRacket: Double = 300.0,
    qPos: Double = 1.0,
    qBounceVx: Double = 20.0,
    qBounceVy: Double = 300.0,
    muInit: DoubleArray? = null,
    m: Array<DoubleArray>? = null,
    perspectiveScale: PerspectiveScale? = null,
) {
    val dt: Double = 1.0 / max(fps, 1e-6)
    val persp: PerspectiveScale = perspectiveScale ?: noPerspective

    var imm: IMMEstimator
    var baseQ: List<Mat>
    var lastDetectionT: Double = t
    var hits = 1
    var lastMeasuredPos: Pair<Double, Double> = Pair(x, y)

    class HistPoint(val t: Double, val x: Double, val y: Double)
    val history = ArrayList<HistPoint>()
    val detTimes = ArrayList<Double>()

    init {
        val f = Mat.from(arrayOf(
            doubleArrayOf(1.0, 0.0, dt, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0, dt),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 0.0, 1.0),
        ))
        val h = Mat.from(arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0, 0.0),
        ))
        val r = Mat.identity(2).scaled(10.0)
        val p0 = Mat.identity(4).scaled(100.0)
        val x0 = Mat.colVec(doubleArrayOf(x, y, vx, vy))

        val qRacketPos = qRacket / 10.0

        fun mk(q: Mat): KalmanFilter =
            KalmanFilter(F = f.clone(), H = h.clone(), R = r.clone(), Q = q,
                P = p0.clone(), x = x0.clone())

        // Model 0 — smooth in-flight CV.
        val kf0 = mk(Mat.diag(doubleArrayOf(qPos, qPos, qSmooth, qSmooth)))
        // Model 1 — racket impact: isotropic high-Q on position AND velocity.
        val kf1 = mk(Mat.diag(doubleArrayOf(qRacketPos, qRacketPos, qRacket, qRacket)))
        // Model 2 — court bounce: anisotropic Q.
        val kf2 = mk(Mat.diag(doubleArrayOf(qPos, qRacketPos, qBounceVx, qBounceVy)))

        val mu = muInit ?: doubleArrayOf(0.90, 0.05, 0.05)
        val trans = m ?: arrayOf(
            doubleArrayOf(0.92, 0.04, 0.04),
            doubleArrayOf(0.70, 0.25, 0.05),
            doubleArrayOf(0.70, 0.05, 0.25),
        )
        imm = IMMEstimator(listOf(kf0, kf1, kf2), mu, trans)
        baseQ = listOf(kf0.Q.clone(), kf1.Q.clone(), kf2.Q.clone())

        history.add(HistPoint(t, x, y))
        detTimes.add(t)
    }

    fun predict() {
        val scale = persp(yPos)
        for (i in imm.filters.indices) {
            imm.filters[i].Q = baseQ[i].scaled(scale)
        }
        imm.predict()
    }

    fun update(x: Double, y: Double, t: Double) {
        imm.update(Mat.colVec(doubleArrayOf(x, y)))
        lastDetectionT = t
        lastMeasuredPos = Pair(x, y)
        hits += 1
        detTimes.add(t)
    }

    fun markMissed() {
        imm.updateWithoutMeasurement()
    }

    val position: Pair<Double, Double> get() = Pair(imm.x.d[0], imm.x.d[1])
    val yPos: Double get() = imm.x.d[1]
    fun speedPxS(): Double = sqrt(imm.x.d[2] * imm.x.d[2] + imm.x.d[3] * imm.x.d[3])
    fun positionUncertainty(): Double = imm.P.at(0, 0) + imm.P.at(1, 1) // trace of P[:2,:2]
    val maneuverProb: Double get() = 1.0 - imm.mu[0]
    val racketProb: Double get() = imm.mu[1]
    val bounceProb: Double get() = imm.mu[2]

    fun record(t: Double, now: Double) {
        val p = position
        history.add(HistPoint(t, p.first, p.second))
        val cutoff = now - motionWindowS
        while (history.isNotEmpty() && history.first().t < cutoff) {
            history.removeAt(0)
        }
        val detCutoff = now - corroborationWindowS
        while (detTimes.isNotEmpty() && detTimes.first() < detCutoff) {
            detTimes.removeAt(0)
        }
    }

    fun recentDetCount(): Int = detTimes.size

    fun recentSpanPx(): Double {
        if (history.size < 2) return 0.0
        val last = history.last()
        var maxD = 0.0
        for (h in history) {
            val d = sqrt((h.x - last.x) * (h.x - last.x) + (h.y - last.y) * (h.y - last.y))
            if (d > maxD) maxD = d
        }
        return maxD
    }

    fun trace(): List<Pair<Double, Double>> = history.map { Pair(it.x, it.y) }
}

private class Tentative(x: Double, y: Double, t: Double) {
    class Pt(val t: Double, val x: Double, val y: Double)
    val points = ArrayList<Pt>()
    var lastT: Double = t

    init {
        points.add(Pt(t, x, y))
    }

    fun add(x: Double, y: Double, t: Double) {
        points.add(Pt(t, x, y))
        lastT = t
    }

    val lastXy: Pair<Double, Double>
        get() = Pair(points[points.size - 1].x, points[points.size - 1].y)

    fun expectedNext(t: Double): Pair<Double, Double> {
        val lx = points[points.size - 1].x
        val ly = points[points.size - 1].y
        if (points.size < 2) return Pair(lx, ly)
        val p0 = points[points.size - 2]
        val p1 = points[points.size - 1]
        val segDt = p1.t - p0.t
        if (segDt <= 0) return Pair(lx, ly)
        val vx = (p1.x - p0.x) / segDt
        val vy = (p1.y - p0.y) / segDt
        val dt = t - p1.t
        return Pair(lx + vx * dt, ly + vy * dt)
    }

    fun spanPx(): Double {
        if (points.size < 2) return 0.0
        val last = points.last()
        var maxD = 0.0
        for (p in points) {
            val d = sqrt((p.x - last.x) * (p.x - last.x) + (p.y - last.y) * (p.y - last.y))
            if (d > maxD) maxD = d
        }
        return maxD
    }

    fun velocity(): Pair<Double, Double> {
        if (points.size < 2) return Pair(0.0, 0.0)
        val p0 = points.first()
        val p1 = points.last()
        val dt = p1.t - p0.t
        if (dt <= 0) return Pair(0.0, 0.0)
        return Pair((p1.x - p0.x) / dt, (p1.y - p0.y) / dt)
    }
}

/**
 * Maintains a single confirmed ball trajectory and reports whether a moving
 * trace is currently alive. Port of ios_tracker BallTrackManager
 * (mobile/lib/engine/ball_tracker.dart, which passes all 10 Python self-test
 * scenarios).
 */
class BallTrackManager(
    val fps: Double,
    perspectiveScale: PerspectiveScale? = null,
    val gateBasePx: Double = 50.0,
    val gateUncertaintyK: Double = 0.6,
    val seedGatePx: Double = 100.0,
    val seedCoherencePx: Double = 38.0,
    val confirmHits: Int = 3,
    val confirmWindowS: Double = 0.6,
    val missTimeoutS: Double = 2.0,
    val motionWindowS: Double = 0.5,
    val moveThreshPx: Double = 30.0,
    val minRecentDets: Int = 3,
    val corroborationWindowS: Double = 2.0,
    val hijackAfterS: Double = 0.15,
    val qSmooth: Double = 5.0,
    val qManeuver: Double = 300.0,
    val qPos: Double = 1.0,
    val qBounceVx: Double = 20.0,
    val qBounceVy: Double = 300.0,
    val fallbackGateK: Double = 1.8,
    val coastGateK: Double = 0.5,
    val coastGateCapPx: Double = 400.0,
) {
    val dt: Double = 1.0 / max(fps, 1e-6)
    private val persp: PerspectiveScale = perspectiveScale ?: noPerspective

    private var track: ConfirmedTrack? = null
    private val tentatives = ArrayList<Tentative>()
    var lastDetectionTime: Double? = null
        private set
    private var lastTrace: List<ConfirmedTrack.HistPoint> = emptyList()

    fun reset() {
        track = null
        tentatives.clear()
        lastDetectionTime = null
        lastTrace = emptyList()
    }

    fun update(detections: List<TrackerDetection>, now: Double): TrackStatus {
        // 1. Predict the confirmed track forward.
        track?.predict()

        // 2. Associate one detection to the confirmed track.
        val used = BooleanArray(detections.size)
        var trackMeasured = false
        val t = track
        if (t != null && detections.isNotEmpty()) {
            val (tx, ty) = t.position
            val scale = persp(t.yPos)
            var gate = gateBasePx * scale +
                gateUncertaintyK * sqrt(max(t.positionUncertainty(), 0.0))
            val tsd = now - t.lastDetectionT
            gate += min(coastGateK * t.speedPxS() * tsd, coastGateCapPx)
            var bestI = -1
            var bestD = gate
            for (i in detections.indices) {
                val d = sqrt((detections[i].x - tx) * (detections[i].x - tx) +
                    (detections[i].y - ty) * (detections[i].y - ty))
                if (d <= bestD) { bestD = d; bestI = i }
            }

            // Fallback gate around the last measured position.
            if (bestI < 0) {
                val (lmx, lmy) = t.lastMeasuredPos
                val fbGate = gateBasePx * fallbackGateK * scale
                var fbBestI = -1
                var fbBestD = fbGate
                for (i in detections.indices) {
                    val d = sqrt((detections[i].x - lmx) * (detections[i].x - lmx) +
                        (detections[i].y - lmy) * (detections[i].y - lmy))
                    if (d <= fbBestD) { fbBestD = d; fbBestI = i }
                }
                bestI = fbBestI
            }

            if (bestI >= 0) {
                t.update(detections[bestI].x, detections[bestI].y, now)
                used[bestI] = true
                lastDetectionTime = now
                trackMeasured = true
            }
        }
        // A frame that gave the track no detection still advances the mode
        // prior one Markov step; otherwise mu stays frozen across a gap.
        if (t != null && !trackMeasured) {
            t.markMissed()
        }

        // 3. Feed leftovers to tentative seeds; try to promote one.
        for (i in detections.indices) {
            if (!used[i]) feedTentative(detections[i].x, detections[i].y, now)
        }
        pruneTentatives(now)
        tryPromote(now)

        // 4. Record position into the motion window.
        track?.record(now, now)

        val status = makeStatus(detections.size, now)
        if (status.state == TrackState.LOST) track = null
        return status
    }

    private fun feedTentative(x: Double, y: Double, now: Double) {
        val scale = persp(y)
        var best: Tentative? = null
        var bestD = Double.POSITIVE_INFINITY
        for (tnt in tentatives) {
            val e = tnt.expectedNext(now)
            val gate = (if (tnt.points.size < 2) seedGatePx else seedCoherencePx) * scale
            val d = sqrt((x - e.first) * (x - e.first) + (y - e.second) * (y - e.second))
            if (d <= gate && d < bestD) { bestD = d; best = tnt }
        }
        if (best != null) {
            best.add(x, y, now)
        } else {
            tentatives.add(Tentative(x, y, now))
        }
    }

    private fun pruneTentatives(now: Double) {
        tentatives.retainAll { now - it.points[0].t <= confirmWindowS }
    }

    private fun tryPromote(now: Double) {
        // The hijack guard protects a *healthy* track from being stolen by a
        // seed. A track sitting on a stationary object (a ball basket, a ball
        // on the ground) is fed a detection every frame, so its lastDetectionT
        // never goes stale and the guard would block promotion forever — the
        // real ball could never take over. Upstream (pipeline/anya_base.py)
        // never hits this because exclusion zones drop stationary ball-like
        // objects before they reach the tracker; this port has no such filter,
        // so only a moving track gets the protection. Promotable seeds must
        // already be coherent moving trajectories, so clutter still can't win.
        val cur = track
        if (cur != null && now - cur.lastDetectionT <= hijackAfterS &&
            cur.recentSpanPx() > moveThreshPx * persp(cur.yPos)) return
        val promotable = tentatives.filter {
            it.points.size >= confirmHits &&
                it.spanPx() > moveThreshPx * persp(it.lastXy.second)
        }
        if (promotable.isEmpty()) return
        val sorted = promotable.sortedWith(Comparator { a, b ->
            if (a.points.size != b.points.size) {
                a.points.size.compareTo(b.points.size)
            } else {
                a.spanPx().compareTo(b.spanPx())
            }
        })
        val best = sorted[sorted.size - 1] // max (len, span)
        val xy = best.lastXy
        val v = best.velocity()
        val tLast = best.points[best.points.size - 1].t
        val nt = ConfirmedTrack(
            fps = fps, x = xy.first, y = xy.second, vx = v.first, vy = v.second, t = tLast,
            motionWindowS = motionWindowS, corroborationWindowS = corroborationWindowS,
            qSmooth = qSmooth, qRacket = qManeuver, qPos = qPos,
            qBounceVx = qBounceVx, qBounceVy = qBounceVy,
            perspectiveScale = persp)
        // Backfill motion window + corroboration from the seed.
        nt.history.clear()
        nt.detTimes.clear()
        for (p in best.points) {
            nt.history.add(ConfirmedTrack.HistPoint(p.t, p.x, p.y))
            nt.detTimes.add(p.t)
        }
        track = nt
        lastDetectionTime = tLast
        tentatives.remove(best)
    }

    private fun makeStatus(ballCount: Int, now: Double): TrackStatus {
        val t = track
        if (t == null) {
            lastTrace = emptyList()
            return TrackStatus(
                hasMovingTrace = false,
                state = TrackState.NONE,
                position = null,
                speedPxS = 0.0,
                timeSinceDetection = 0.0,
                coasting = false,
                ballCount = ballCount,
                maneuverProb = 0.0,
                racketProb = 0.0,
                bounceProb = 0.0,
                trace = emptyList())
        }

        val tsd = now - t.lastDetectionT
        val lost = tsd > missTimeoutS
        val coasting = !lost && tsd > (1.5 * dt)
        val moving = t.recentSpanPx() > moveThreshPx * persp(t.yPos)
        val corroborated = t.recentDetCount() >= minRecentDets

        lastTrace = ArrayList(t.history)

        val state: TrackState
        val alive: Boolean
        when {
            lost -> { state = TrackState.LOST; alive = false }
            !corroborated -> { state = TrackState.FADING; alive = false }
            !moving -> { state = TrackState.STOPPED; alive = false }
            else -> { state = if (coasting) TrackState.COASTING else TrackState.MOVING; alive = true }
        }

        return TrackStatus(
            hasMovingTrace = alive,
            state = state,
            position = t.position,
            speedPxS = t.speedPxS(),
            timeSinceDetection = tsd,
            coasting = coasting,
            ballCount = ballCount,
            maneuverProb = t.maneuverProb,
            racketProb = t.racketProb,
            bounceProb = t.bounceProb,
            trace = t.trace())
    }

    /** (t, x, y) points of the current motion-window trace. */
    fun tracePoints(): List<Triple<Double, Double, Double>> =
        lastTrace.map { Triple(it.t, it.x, it.y) }
}
