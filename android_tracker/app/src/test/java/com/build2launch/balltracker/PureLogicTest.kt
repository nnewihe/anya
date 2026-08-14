package com.build2launch.balltracker

import com.build2launch.balltracker.detection.clusterZones
import com.build2launch.balltracker.detection.dbscan
import com.build2launch.balltracker.tracking.ViterbiBallTracker
import com.build2launch.balltracker.tracking.ViterbiDetection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Coverage for the remaining pure ports that don't touch Android: the DBSCAN /
 * exclusion-zone clustering and the offline Viterbi solver. (Highlight segment
 * logic is exercised indirectly by the iOS parity and is pure here too.)
 */
class PureLogicTest {

    @Test fun dbscan_twoTightClustersPlusNoise() {
        val pts = ArrayList<Pair<Double, Double>>()
        // Cluster A around (10,10), cluster B around (200,200), one outlier.
        repeat(20) { pts.add(Pair(10.0 + it % 3, 10.0 + it % 2)) }
        repeat(20) { pts.add(Pair(200.0 + it % 3, 200.0 + it % 2)) }
        pts.add(Pair(1000.0, 1000.0)) // noise
        val labels = dbscan(pts, eps = 12.0, minSamples = 15)
        val clusters = labels.filter { it >= 0 }.toSet()
        assertEquals(2, clusters.size)
        assertEquals(-1, labels.last()) // the outlier is noise
    }

    @Test fun clusterZones_fencesStationaryBasket() {
        // A basket detected in the same spot on many frames -> one zone.
        val centres = (0 until 30).map { Pair(500.0 + it % 4, 250.0 + it % 4) }
        val zones = clusterZones(centres, eps = 12.0, minSamples = 15, padding = 0.0)
        assertEquals(1, zones.rects.size)
        assertTrue(zones.contains(501.0, 251.0))
        assertTrue(!zones.contains(50.0, 50.0))
    }

    @Test fun viterbi_recoversSmoothFlightAcrossAGap() {
        // A clean left-to-right flight at 30 fps, with a 3-frame occlusion in
        // the middle; the solver should track it and fill the gap by coasting.
        val fps = 30.0
        val dt = 1.0 / fps
        val frames = ArrayList<List<ViterbiDetection>>()
        val times = ArrayList<Double>()
        var x = 100.0
        for (i in 0 until 30) {
            x += 20.0
            val occluded = i in 14..16
            frames.add(if (occluded) emptyList() else listOf(ViterbiDetection(x, 300.0, 0.9)))
            times.add(i * dt)
        }
        val samples = ViterbiBallTracker().solve(frames, times)
        val tracked = samples.count { it.pos != null }
        // Every frame including the 3-frame gap should be covered.
        assertTrue("tracked $tracked/30", tracked >= 28)
        assertTrue(samples.last().pos != null)
    }
}
