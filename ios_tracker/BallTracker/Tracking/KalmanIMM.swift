import Foundation

/// Port of filterpy's `KalmanFilter` for the fixed shape used by the ball
/// tracker (dim_x=4, dim_z=2). Same predict/update equations, and the same
/// multivariate-normal `likelihood` used by the IMM.
/// Port of mobile/lib/engine/kalman.dart.
final class KalmanFilter {
    var F: Mat
    var H: Mat
    var R: Mat
    var Q: Mat
    var P: Mat
    var x: Mat
    var y: Mat = Mat(2, 1)      // residual
    var S: Mat = Mat(2, 2)      // system uncertainty
    var logLikelihood = 0.0
    var likelihood = 1.0

    init(F: Mat, H: Mat, R: Mat, Q: Mat, P: Mat, x: Mat) {
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q
        self.P = P
        self.x = x
    }

    func predict() {
        x = F.matmul(x)
        P = F.matmul(P).matmul(F.transpose()) + Q
    }

    func update(_ z: Mat) {
        y = z - H.matmul(x)
        let ht = H.transpose()
        let pht = P.matmul(ht)
        S = H.matmul(pht) + R
        let k = pht.matmul(S.inverse())
        x = x + k.matmul(y)
        let iKh = Mat.identity(x.rows) - k.matmul(H)
        P = iKh.matmul(P).matmul(iKh.transpose()) +
            k.matmul(R).matmul(k.transpose())
        logLikelihood = logpdf(y, S)
        likelihood = exp(logLikelihood)
        if likelihood == 0.0 { likelihood = 5e-324 } // smallest positive double
    }
}

/// Port of filterpy's `IMMEstimator`. Interacting-multiple-model mixing over a
/// set of Kalman filters with mode probabilities `mu` and Markov transition
/// matrix `M`.
final class IMMEstimator {
    let filters: [KalmanFilter]
    let n: Int
    var mu: [Double]
    let M: [[Double]]
    var cbar: [Double] = []
    var omega: [[Double]]
    var x: Mat = Mat(4, 1)
    var P: Mat = Mat(4, 4)

    init(_ filters: [KalmanFilter], _ muInit: [Double], _ M: [[Double]]) {
        self.filters = filters
        self.n = filters.count
        let s = muInit.reduce(0.0, +)
        self.mu = muInit.map { $0 / s }
        self.M = M
        self.omega = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        computeMixingProbabilities()
        computeStateEstimate()
    }

    func predict() {
        let dim = filters[0].x.rows
        var xs: [Mat] = []
        var ps: [Mat] = []
        // Mixed initial conditions for each target filter i using column omega[:, i].
        for i in 0..<n {
            let xMix = Mat(dim, 1)
            for j in 0..<n {
                let w = omega[j][i]
                for r in 0..<dim {
                    xMix.d[r] += filters[j].x.d[r] * w
                }
            }
            xs.append(xMix)
        }
        for i in 0..<n {
            var pMix = Mat(dim, dim)
            for j in 0..<n {
                let w = omega[j][i]
                let yv = filters[j].x - xs[i]
                pMix = pMix + (yv.outer(yv) + filters[j].P).scaled(w)
            }
            ps.append(pMix)
        }
        for i in 0..<n {
            filters[i].x = xs[i].clone()
            filters[i].P = ps[i].clone()
            filters[i].predict()
        }
        computeStateEstimate()
    }

    func update(_ z: Mat) {
        var likelihood = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            filters[i].update(z)
            likelihood[i] = filters[i].likelihood
        }
        var sum = 0.0
        var newMu = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            newMu[i] = cbar[i] * likelihood[i]
            sum += newMu[i]
        }
        for i in 0..<n { newMu[i] /= sum }
        mu = newMu
        computeMixingProbabilities()
        computeStateEstimate()
    }

    private func computeMixingProbabilities() {
        // cbar[j] = sum_i mu[i] * M[i][j]
        cbar = [Double](repeating: 0.0, count: n)
        for j in 0..<n {
            var s = 0.0
            for i in 0..<n { s += mu[i] * M[i][j] }
            cbar[j] = s
        }
        for i in 0..<n {
            for j in 0..<n {
                omega[i][j] = (M[i][j] * mu[i]) / cbar[j]
            }
        }
    }

    private func computeStateEstimate() {
        let dim = filters[0].x.rows
        x = Mat(dim, 1)
        for i in 0..<n {
            for r in 0..<dim {
                x.d[r] += filters[i].x.d[r] * mu[i]
            }
        }
        P = Mat(dim, dim)
        for i in 0..<n {
            let yv = filters[i].x - x
            P = P + (yv.outer(yv) + filters[i].P).scaled(mu[i])
        }
    }
}
