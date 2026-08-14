package com.build2launch.balltracker.tracking

import kotlin.math.exp

/**
 * Port of filterpy's `KalmanFilter` for the fixed shape used by the ball
 * tracker (dim_x=4, dim_z=2). Same predict/update equations, and the same
 * multivariate-normal `likelihood` used by the IMM.
 * Port of ios_tracker BallTracker/Tracking/KalmanIMM.swift
 * (mobile/lib/engine/kalman.dart).
 */
class KalmanFilter(
    var F: Mat,
    var H: Mat,
    var R: Mat,
    var Q: Mat,
    var P: Mat,
    var x: Mat,
) {
    var y: Mat = Mat(2, 1)   // residual
    var S: Mat = Mat(2, 2)   // system uncertainty
    var logLikelihood = 0.0
    var likelihood = 1.0

    fun predict() {
        x = F.matmul(x)
        P = F.matmul(P).matmul(F.transpose()) + Q
    }

    fun update(z: Mat) {
        y = z - H.matmul(x)
        val ht = H.transpose()
        val pht = P.matmul(ht)
        S = H.matmul(pht) + R
        val k = pht.matmul(S.inverse())
        x = x + k.matmul(y)
        val iKh = Mat.identity(x.rows) - k.matmul(H)
        P = iKh.matmul(P).matmul(iKh.transpose()) +
            k.matmul(R).matmul(k.transpose())
        logLikelihood = logpdf(y, S)
        likelihood = exp(logLikelihood)
        if (likelihood == 0.0) likelihood = 5e-324 // smallest positive double
    }
}

/**
 * Port of filterpy's `IMMEstimator`. Interacting-multiple-model mixing over a
 * set of Kalman filters with mode probabilities `mu` and Markov transition
 * matrix `M`.
 */
class IMMEstimator(
    val filters: List<KalmanFilter>,
    muInit: DoubleArray,
    val M: Array<DoubleArray>,
) {
    val n: Int = filters.size
    var mu: DoubleArray
    var cbar: DoubleArray = DoubleArray(0)
    var omega: Array<DoubleArray>
    var x: Mat = Mat(4, 1)
    var P: Mat = Mat(4, 4)

    init {
        val s = muInit.sum()
        mu = DoubleArray(n) { muInit[it] / s }
        omega = Array(n) { DoubleArray(n) }
        computeMixingProbabilities()
        computeStateEstimate()
    }

    fun predict() {
        val dim = filters[0].x.rows
        val xs = ArrayList<Mat>(n)
        val ps = ArrayList<Mat>(n)
        // Mixed initial conditions for each target filter i using column omega[:, i].
        for (i in 0 until n) {
            val xMix = Mat(dim, 1)
            for (j in 0 until n) {
                val w = omega[j][i]
                for (r in 0 until dim) {
                    xMix.d[r] += filters[j].x.d[r] * w
                }
            }
            xs.add(xMix)
        }
        for (i in 0 until n) {
            var pMix = Mat(dim, dim)
            for (j in 0 until n) {
                val w = omega[j][i]
                val yv = filters[j].x - xs[i]
                pMix += (yv.outer(yv) + filters[j].P).scaled(w)
            }
            ps.add(pMix)
        }
        for (i in 0 until n) {
            filters[i].x = xs[i].clone()
            filters[i].P = ps[i].clone()
            filters[i].predict()
        }
        computeStateEstimate()
    }

    fun update(z: Mat) {
        // Mode probabilities are updated in log-space (deviating from
        // filterpy, which multiplies raw likelihoods). After a long coast the
        // innovation can be large enough that every filter's likelihood
        // underflows to 0, and the raw product degenerates to mu = cbar — the
        // measurement is thrown away exactly when it carries the most
        // information (e.g. the ball reappearing with reversed velocity after
        // an unseen racket hit). Shifting by the max log-likelihood keeps the
        // ratios, which is all the normalisation needs.
        val logL = DoubleArray(n)
        for (i in 0 until n) {
            filters[i].update(z)
            logL[i] = filters[i].logLikelihood
        }
        val maxLog = logL.max()
        var sum = 0.0
        val newMu = DoubleArray(n)
        for (i in 0 until n) {
            newMu[i] = cbar[i] * (if (maxLog.isFinite()) exp(logL[i] - maxLog) else 1.0)
            sum += newMu[i]
        }
        for (i in 0 until n) newMu[i] /= sum
        mu = newMu
        computeMixingProbabilities()
        computeStateEstimate()
    }

    /**
     * Mode-probability time update for a frame with no measurement.
     * `update()` is the only place `mu` moves, and it folds in exactly one
     * step of the Markov chain — so during an occlusion the mode prior stays
     * frozen and a k-frame gap gets priced as a single transition step.
     * Advancing `mu <- mu·M` (= `cbar`) once per missed frame makes the prior
     * for a hidden racket hit or bounce reflect elapsed unobserved time,
     * relaxing toward the chain's stationary distribution.
     */
    fun updateWithoutMeasurement() {
        mu = cbar.copyOf()
        computeMixingProbabilities()
        computeStateEstimate()
    }

    private fun computeMixingProbabilities() {
        // cbar[j] = sum_i mu[i] * M[i][j]
        cbar = DoubleArray(n)
        for (j in 0 until n) {
            var s = 0.0
            for (i in 0 until n) s += mu[i] * M[i][j]
            cbar[j] = s
        }
        for (i in 0 until n) {
            for (j in 0 until n) {
                omega[i][j] = (M[i][j] * mu[i]) / cbar[j]
            }
        }
    }

    private fun computeStateEstimate() {
        val dim = filters[0].x.rows
        x = Mat(dim, 1)
        for (i in 0 until n) {
            for (r in 0 until dim) {
                x.d[r] += filters[i].x.d[r] * mu[i]
            }
        }
        P = Mat(dim, dim)
        for (i in 0 until n) {
            val yv = filters[i].x - x
            P += (yv.outer(yv) + filters[i].P).scaled(mu[i])
        }
    }
}
