package com.build2launch.balltracker.detection

/**
 * Axis-aligned rectangle in analysis space (doubles, no android.graphics
 * dependency so the zone logic stays JVM-unit-testable).
 */
data class Zone(val minX: Double, val minY: Double, val maxX: Double, val maxY: Double)

/**
 * Rectangles (in 960-wide analysis space) fencing off stationary ball-like
 * clutter — ball baskets, balls sitting on the court. Detections whose centre
 * lands inside one are dropped before they reach the tracker.
 *
 * Port of `_is_in_exclusion_zone` / `create_auto_exclusion_zones` from
 * pipeline/utilities.py (via ios_tracker ExclusionZones.swift). This matters
 * more than it looks: a ball basket is detected at ~0.8 conf on *every* frame,
 * so a track that latches onto it is never starved of detections, never times
 * out, and blocks the real ball from ever being promoted. Fencing the basket
 * off is what keeps the tracker honest.
 */
class ExclusionZones(val rects: List<Zone>) {
    val isEmpty: Boolean get() = rects.isEmpty()

    /** Inclusive on all edges, matching the Python `x1 <= x <= x2` test. */
    fun contains(x: Double, y: Double): Boolean {
        for (r in rects) {
            if (x >= r.minX && x <= r.maxX && y >= r.minY && y <= r.maxY) return true
        }
        return false
    }

    companion object {
        val NONE = ExclusionZones(emptyList())
    }
}

/**
 * DBSCAN over 2-D points, matching scikit-learn's semantics: a point's
 * eps-neighbourhood includes the point itself, so `minSamples` counts it.
 * Returns one label per point; -1 means noise.
 */
fun dbscan(pts: List<Pair<Double, Double>>, eps: Double, minSamples: Int): IntArray {
    val unvisited = -2
    val noise = -1
    val labels = IntArray(pts.size) { unvisited }
    var cluster = 0
    val eps2 = eps * eps

    fun neighbours(i: Int): ArrayList<Int> {
        val out = ArrayList<Int>()
        for (j in pts.indices) {
            val dx = pts[i].first - pts[j].first
            val dy = pts[i].second - pts[j].second
            if (dx * dx + dy * dy <= eps2) out.add(j)
        }
        return out
    }

    for (i in pts.indices) {
        if (labels[i] != unvisited) continue
        val seeds = neighbours(i)
        if (seeds.size < minSamples) {
            labels[i] = noise      // may still be claimed later as a border point
            continue
        }
        labels[i] = cluster

        val queue = ArrayList(seeds.filter { it != i })
        val queued = BooleanArray(pts.size)
        queued[i] = true
        for (s in queue) queued[s] = true

        var k = 0
        while (k < queue.size) {
            val j = queue[k]
            k += 1
            if (labels[j] == noise) labels[j] = cluster   // border point
            if (labels[j] != unvisited) continue
            labels[j] = cluster
            val jn = neighbours(j)
            if (jn.size >= minSamples) {                   // core point — expand
                for (m in jn) {
                    if (!queued[m]) { queued[m] = true; queue.add(m) }
                }
            }
        }
        cluster += 1
    }
    return labels
}

/**
 * Cluster centres that keep landing in the same place into exclusion rects.
 * Pure counterpart of ExclusionZoneScanner.scan's clustering step — the frame
 * sampling + detection lives in ExclusionZoneScanner (Android), this is the
 * part worth unit testing.
 */
fun clusterZones(centres: List<Pair<Double, Double>>, eps: Double, minSamples: Int,
                 padding: Double): ExclusionZones {
    if (centres.size < minSamples) return ExclusionZones.NONE
    val labels = dbscan(centres, eps, minSamples)
    val rects = ArrayList<Zone>()
    for (kk in labels.toSet()) {
        if (kk < 0) continue
        val pts = centres.filterIndexed { idx, _ -> labels[idx] == kk }
        if (pts.isEmpty()) continue
        var minX = pts[0].first; var maxX = pts[0].first
        var minY = pts[0].second; var maxY = pts[0].second
        for (p in pts) {
            minX = minOf(minX, p.first); maxX = maxOf(maxX, p.first)
            minY = minOf(minY, p.second); maxY = maxOf(maxY, p.second)
        }
        rects.add(Zone(minX - padding, minY - padding, maxX + padding, maxY + padding))
    }
    return ExclusionZones(rects)
}
