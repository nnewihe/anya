package com.build2launch.balltracker.tracking

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.PI

/**
 * Minimal dense matrix for the Kalman / IMM port. Row-major storage; only the
 * operations the ball tracker needs, matching numpy/filterpy semantics.
 * Port of ios_tracker BallTracker/Tracking/Matrix.swift, itself a port of
 * mobile/lib/engine/linalg.dart (which passes the Python parity tests).
 */
class Mat {
    val rows: Int
    val cols: Int
    val d: DoubleArray // row-major, length rows*cols

    constructor(rows: Int, cols: Int) {
        this.rows = rows
        this.cols = cols
        this.d = DoubleArray(rows * cols)
    }

    constructor(rows: Int, cols: Int, d: DoubleArray) {
        this.rows = rows
        this.cols = cols
        this.d = d
    }

    fun at(r: Int, c: Int): Double = d[r * cols + c]
    fun set(r: Int, c: Int, v: Double) { d[r * cols + c] = v }

    fun clone(): Mat = Mat(rows, cols, d.copyOf())

    fun matmul(b: Mat): Mat {
        require(cols == b.rows)
        val out = Mat(rows, b.cols)
        for (i in 0 until rows) {
            for (k in 0 until cols) {
                val a = d[i * cols + k]
                if (a == 0.0) continue
                for (j in 0 until b.cols) {
                    out.d[i * b.cols + j] += a * b.d[k * b.cols + j]
                }
            }
        }
        return out
    }

    fun transpose(): Mat {
        val out = Mat(cols, rows)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                out.d[j * rows + i] = d[i * cols + j]
            }
        }
        return out
    }

    operator fun plus(b: Mat): Mat {
        val out = Mat(rows, cols)
        for (i in d.indices) out.d[i] = d[i] + b.d[i]
        return out
    }

    operator fun minus(b: Mat): Mat {
        val out = Mat(rows, cols)
        for (i in d.indices) out.d[i] = d[i] - b.d[i]
        return out
    }

    fun scaled(s: Double): Mat {
        val out = Mat(rows, cols)
        for (i in d.indices) out.d[i] = d[i] * s
        return out
    }

    /** Outer product of two column vectors (self·otherᵀ), both n×1. */
    fun outer(other: Mat): Mat {
        val out = Mat(rows, other.rows)
        for (i in 0 until rows) {
            for (j in 0 until other.rows) {
                out.d[i * other.rows + j] = d[i] * other.d[j]
            }
        }
        return out
    }

    /** General square-matrix inverse via Gauss–Jordan with partial pivoting. */
    fun inverse(): Mat {
        require(rows == cols)
        val n = rows
        val a = DoubleArray(n * 2 * n)
        for (i in 0 until n) {
            for (j in 0 until n) a[i * 2 * n + j] = d[i * n + j]
            a[i * 2 * n + n + i] = 1.0
        }
        for (col in 0 until n) {
            var piv = col
            var best = abs(a[col * 2 * n + col])
            for (r in (col + 1) until n) {
                val v = abs(a[r * 2 * n + col])
                if (v > best) { best = v; piv = r }
            }
            if (piv != col) {
                for (j in 0 until (2 * n)) {
                    val tmp = a[col * 2 * n + j]
                    a[col * 2 * n + j] = a[piv * 2 * n + j]
                    a[piv * 2 * n + j] = tmp
                }
            }
            val pivVal = a[col * 2 * n + col]
            if (abs(pivVal) < 1e-12) {
                continue // singular-ish; leave row (mirrors the Dart port)
            }
            val inv = 1.0 / pivVal
            for (j in 0 until (2 * n)) a[col * 2 * n + j] *= inv
            for (r in 0 until n) {
                if (r == col) continue
                val f = a[r * 2 * n + col]
                if (f == 0.0) continue
                for (j in 0 until (2 * n)) {
                    a[r * 2 * n + j] -= f * a[col * 2 * n + j]
                }
            }
        }
        val out = Mat(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) out.d[i * n + j] = a[i * 2 * n + n + j]
        }
        return out
    }

    fun det2x2(): Double = d[0] * d[3] - d[1] * d[2]

    companion object {
        fun identity(n: Int): Mat {
            val m = Mat(n, n)
            for (i in 0 until n) m.d[i * n + i] = 1.0
            return m
        }

        fun colVec(v: DoubleArray): Mat = Mat(v.size, 1, v.copyOf())

        fun diag(v: DoubleArray): Mat {
            val n = v.size
            val m = Mat(n, n)
            for (i in 0 until n) m.d[i * n + i] = v[i]
            return m
        }

        fun from(r: Array<DoubleArray>): Mat {
            val flat = DoubleArray(r.size * r[0].size)
            var k = 0
            for (row in r) for (x in row) flat[k++] = x
            return Mat(r.size, r[0].size, flat)
        }
    }
}

/**
 * Log-pdf of a zero-mean multivariate normal at residual `y` (n×1) with
 * covariance `s` (n×n). Mirrors filterpy's `logpdf` used for IMM likelihoods.
 */
fun logpdf(y: Mat, s: Mat): Double {
    val n = y.rows
    val sInv = s.inverse()
    val tmp = sInv.matmul(y)
    var quad = 0.0
    for (i in 0 until n) quad += y.d[i] * tmp.d[i]
    val detS = if (n == 2) s.det2x2() else det(s)
    val logDet = ln(minOf(maxOf(abs(detS), 1e-300), Double.MAX_VALUE))
    return -0.5 * (n.toDouble() * ln(2 * PI) + logDet + quad)
}

private fun det(m: Mat): Double {
    val n = m.rows
    val a = m.d.copyOf()
    var det = 1.0
    for (col in 0 until n) {
        var piv = col
        var best = abs(a[col * n + col])
        for (r in (col + 1) until n) {
            val v = abs(a[r * n + col])
            if (v > best) { best = v; piv = r }
        }
        if (piv != col) {
            for (j in 0 until n) {
                val tmp = a[col * n + j]; a[col * n + j] = a[piv * n + j]; a[piv * n + j] = tmp
            }
            det = -det
        }
        val pv = a[col * n + col]
        if (abs(pv) < 1e-300) return 0.0
        det *= pv
        for (r in (col + 1) until n) {
            val f = a[r * n + col] / pv
            for (j in col until n) a[r * n + j] -= f * a[col * n + j]
        }
    }
    return det
}
