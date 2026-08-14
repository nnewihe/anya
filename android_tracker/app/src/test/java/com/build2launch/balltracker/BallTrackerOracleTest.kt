package com.build2launch.balltracker

import com.build2launch.balltracker.tracking.BallTrackManager
import com.build2launch.balltracker.tracking.PerspectiveScale
import com.build2launch.balltracker.tracking.TrackState
import com.build2launch.balltracker.tracking.TrackStatus
import com.build2launch.balltracker.tracking.TrackerDetection
import com.build2launch.balltracker.tracking.makeImageRowPerspective
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

/**
 * Tracker parity: the 10 self-test scenarios from pipeline/ball_tracker.py /
 * mobile/test/ball_tracker_test.dart, ported straight from the iOS parity
 * harness (ios_tracker/ParityCheck/main.swift). These exercise the shipped
 * Kalman/IMM/BallTrackManager code on the JVM — no model or device needed.
 */
class BallTrackerOracleTest {
    private val fps = 30.0
    private val dt = 1.0 / fps

    private fun run(stream: List<List<TrackerDetection>>, persp: PerspectiveScale? = null): List<TrackStatus> {
        val mgr = BallTrackManager(fps = fps, perspectiveScale = persp)
        val out = ArrayList<TrackStatus>()
        var t = 0.0
        for (dets in stream) {
            out.add(mgr.update(dets, t))
            t += dt
        }
        return out
    }

    private fun d1(x: Double, y: Double, c: Double): List<TrackerDetection> =
        listOf(TrackerDetection(x, y, c))

    @Test fun s1_movingBallLiveTrace() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 100.0
        repeat(40) { x += 18.0; stream.add(d1(x, 300.0, 0.9)) }
        val res = run(stream)
        assertTrue(res.any { it.hasMovingTrace } && res.last().hasMovingTrace &&
            res.last().state == TrackState.MOVING)
    }

    @Test fun s2_stoppedBallEndsTrace() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 100.0
        repeat(25) { x += 18.0; stream.add(d1(x, 300.0, 0.9)) }
        repeat(25) { stream.add(d1(x, 300.0, 0.9)) }
        val res = run(stream)
        assertTrue(!res.last().hasMovingTrace && res.last().state == TrackState.STOPPED)
    }

    @Test fun s3_lostWithinTimeout() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 100.0
        repeat(25) { x += 18.0; stream.add(d1(x, 300.0, 0.9)) }
        repeat(100) { stream.add(emptyList()) }
        val res = run(stream)
        val aliveIdx = res.indices.filter { res[it].hasMovingTrace }
        val lastAlive = aliveIdx.lastOrNull() ?: 24
        val coastS = (lastAlive - 24) * dt
        assertTrue(!res.last().hasMovingTrace && coastS <= 2.0 + dt + 1e-9 && lastAlive >= 24)
    }

    @Test fun s4_scatteredFalsePositives() {
        var seed = 0x9E3779B97F4A7C15uL
        fun nextRand(): Double {
            seed = seed * 6364136223846793005uL + 1442695040888963407uL
            return (seed shr 11).toDouble() / (1UL shl 53).toDouble()
        }
        val stream = ArrayList<List<TrackerDetection>>()
        repeat(40) {
            if (nextRand() < 0.4) {
                stream.add(d1(Math.floor(nextRand() * 900), Math.floor(nextRand() * 500), 0.3))
            } else {
                stream.add(emptyList())
            }
        }
        val res = run(stream)
        assertTrue(res.none { it.hasMovingTrace })
    }

    @Test fun s5_stationaryBallNoTrace() {
        val stream = List(40) { d1(500.0, 250.0, 0.8) }
        val res = run(stream)
        assertTrue(res.none { it.hasMovingTrace })
    }

    @Test fun s6_survivesOcclusion() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 100.0
        repeat(20) { x += 18.0; stream.add(d1(x, 300.0, 0.9)) }
        repeat(6) { x += 18.0; stream.add(emptyList()) }
        repeat(20) { x += 18.0; stream.add(d1(x, 300.0, 0.9)) }
        val res = run(stream)
        assertTrue(res.last().hasMovingTrace && res.subList(25, res.size).all { it.hasMovingTrace })
    }

    @Test fun s7_racketReversal() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 200.0
        repeat(25) { x += 30.0; stream.add(d1(x, 300.0, 0.9)) }
        stream.add(emptyList())
        repeat(25) { x -= 30.0; stream.add(d1(x, 300.0, 0.9)) }
        val res = run(stream)
        val aliveAfter = res.subList(26, res.size).map { it.hasMovingTrace }
        var maxDead = 0; var cur = 0
        for (a in aliveAfter) { cur = if (a) 0 else cur + 1; maxDead = maxOf(maxDead, cur) }
        val mp = res.subList(24, 34).maxOf { it.maneuverProb }
        val rp = res.subList(24, 34).maxOf { it.racketProb }
        val bp = res.subList(24, 34).maxOf { it.bounceProb }
        assertTrue(aliveAfter.takeLast(10).all { it } && maxDead <= 2 && mp > 0.5 && rp > bp)
    }

    @Test fun s8_courtBounce() {
        val stream = ArrayList<List<TrackerDetection>>()
        var x = 200.0; var y = 100.0
        repeat(25) { x += 15.0; y += 20.0; stream.add(d1(x, y, 0.9)) }
        repeat(25) { x += 15.0; y -= 20.0; stream.add(d1(x, y, 0.9)) }
        val res = run(stream)
        val aliveAfter = res.subList(25, res.size).map { it.hasMovingTrace }
        var maxDead = 0; var cur = 0
        for (a in aliveAfter) { cur = if (a) 0 else cur + 1; maxDead = maxOf(maxDead, cur) }
        val mp = res.subList(23, 33).maxOf { it.maneuverProb }
        val rp = res.subList(23, 40).maxOf { it.racketProb }
        val bp = res.subList(23, 40).maxOf { it.bounceProb }
        assertTrue(aliveAfter.takeLast(10).all { it } && maxDead <= 2 && mp > 0.5 && bp > rp)
    }

    @Test fun s9_serveReacquire() {
        val stream = ArrayList<List<TrackerDetection>>()
        val xt = 400.0
        var yt = 400.0
        repeat(20) { yt -= 20.0; stream.add(d1(xt, yt, 0.9)) }
        stream.add(emptyList())
        var xs = xt
        val ys = yt
        repeat(40) { xs += 100.0; stream.add(d1(xs, ys, 0.9)) }
        val res = run(stream)
        val alivePost = res.subList(22, res.size).map { it.hasMovingTrace }
        assertTrue(alivePost.contains(true) && res.last().hasMovingTrace)
    }

    @Test fun s10_sparseNetCrossing() {
        val persp = makeImageRowPerspective(frameHeight = 540.0)
        val mgr = BallTrackManager(fps = fps, perspectiveScale = persp)
        val stream = ArrayList<List<TrackerDetection>>()
        val truth = ArrayList<Pair<Double, Double>>()
        var xn = 150.0; var yn = 460.0
        repeat(16) {
            xn += 26.0; yn -= 11.0
            stream.add(d1(xn, yn, 0.9)); truth.add(Pair(xn, yn))
        }
        repeat(7) {
            xn += 15.0; yn -= 6.0
            stream.add(emptyList()); truth.add(Pair(xn, yn))
        }
        repeat(28) { i ->
            xn += 15.0; yn -= 6.0
            stream.add(if (i % 4 == 0) d1(xn, yn, 0.8) else emptyList()); truth.add(Pair(xn, yn))
        }
        val errs = ArrayList<Double>()
        var lastStatus: TrackStatus? = null
        var tt = 0.0
        for (i in stream.indices) {
            val s = mgr.update(stream[i], tt)
            tt += dt
            lastStatus = s
            val p = s.position
            if (p != null) {
                errs.add(sqrt((p.first - truth[i].first) * (p.first - truth[i].first) +
                    (p.second - truth[i].second) * (p.second - truth[i].second)))
            }
        }
        assertTrue(errs.last() < 60.0 && errs.subList(20, errs.size).max() < 120.0 &&
            lastStatus!!.hasMovingTrace)
    }
}
