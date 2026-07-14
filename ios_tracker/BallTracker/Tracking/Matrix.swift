import Foundation

/// Minimal dense matrix for the Kalman / IMM port. Row-major storage; only the
/// operations the ball tracker needs, matching numpy/filterpy semantics.
/// Port of mobile/lib/engine/linalg.dart (which passes the Python parity tests).
final class Mat {
    let rows: Int
    let cols: Int
    var d: [Double] // row-major, length rows*cols

    init(_ rows: Int, _ cols: Int) {
        self.rows = rows
        self.cols = cols
        self.d = [Double](repeating: 0.0, count: rows * cols)
    }

    init(rows: Int, cols: Int, d: [Double]) {
        self.rows = rows
        self.cols = cols
        self.d = d
    }

    convenience init(from r: [[Double]]) {
        self.init(rows: r.count, cols: r[0].count, d: r.flatMap { $0 })
    }

    static func identity(_ n: Int) -> Mat {
        let m = Mat(n, n)
        for i in 0..<n { m.d[i * n + i] = 1.0 }
        return m
    }

    static func colVec(_ v: [Double]) -> Mat {
        Mat(rows: v.count, cols: 1, d: v)
    }

    static func diag(_ v: [Double]) -> Mat {
        let n = v.count
        let m = Mat(n, n)
        for i in 0..<n { m.d[i * n + i] = v[i] }
        return m
    }

    func at(_ r: Int, _ c: Int) -> Double { d[r * cols + c] }
    func set(_ r: Int, _ c: Int, _ v: Double) { d[r * cols + c] = v }

    func clone() -> Mat { Mat(rows: rows, cols: cols, d: d) }

    func matmul(_ b: Mat) -> Mat {
        precondition(cols == b.rows)
        let out = Mat(rows, b.cols)
        for i in 0..<rows {
            for k in 0..<cols {
                let a = d[i * cols + k]
                if a == 0.0 { continue }
                for j in 0..<b.cols {
                    out.d[i * b.cols + j] += a * b.d[k * b.cols + j]
                }
            }
        }
        return out
    }

    func transpose() -> Mat {
        let out = Mat(cols, rows)
        for i in 0..<rows {
            for j in 0..<cols {
                out.d[j * rows + i] = d[i * cols + j]
            }
        }
        return out
    }

    static func + (a: Mat, b: Mat) -> Mat {
        let out = Mat(a.rows, a.cols)
        for i in 0..<a.d.count { out.d[i] = a.d[i] + b.d[i] }
        return out
    }

    static func - (a: Mat, b: Mat) -> Mat {
        let out = Mat(a.rows, a.cols)
        for i in 0..<a.d.count { out.d[i] = a.d[i] - b.d[i] }
        return out
    }

    func scaled(_ s: Double) -> Mat {
        let out = Mat(rows, cols)
        for i in 0..<d.count { out.d[i] = d[i] * s }
        return out
    }

    /// Outer product of two column vectors (self·otherᵀ), both n×1.
    func outer(_ other: Mat) -> Mat {
        let out = Mat(rows, other.rows)
        for i in 0..<rows {
            for j in 0..<other.rows {
                out.d[i * other.rows + j] = d[i] * other.d[j]
            }
        }
        return out
    }

    /// General square-matrix inverse via Gauss–Jordan with partial pivoting.
    func inverse() -> Mat {
        precondition(rows == cols)
        let n = rows
        var a = [Double](repeating: 0.0, count: n * 2 * n)
        for i in 0..<n {
            for j in 0..<n { a[i * 2 * n + j] = d[i * n + j] }
            a[i * 2 * n + n + i] = 1.0
        }
        for col in 0..<n {
            var piv = col
            var best = abs(a[col * 2 * n + col])
            for r in (col + 1)..<n {
                let v = abs(a[r * 2 * n + col])
                if v > best { best = v; piv = r }
            }
            if piv != col {
                for j in 0..<(2 * n) {
                    a.swapAt(col * 2 * n + j, piv * 2 * n + j)
                }
            }
            let pivVal = a[col * 2 * n + col]
            if abs(pivVal) < 1e-12 {
                continue // singular-ish; leave row (mirrors the Dart port)
            }
            let inv = 1.0 / pivVal
            for j in 0..<(2 * n) { a[col * 2 * n + j] *= inv }
            for r in 0..<n where r != col {
                let f = a[r * 2 * n + col]
                if f == 0.0 { continue }
                for j in 0..<(2 * n) {
                    a[r * 2 * n + j] -= f * a[col * 2 * n + j]
                }
            }
        }
        let out = Mat(n, n)
        for i in 0..<n {
            for j in 0..<n { out.d[i * n + j] = a[i * 2 * n + n + j] }
        }
        return out
    }

    func det2x2() -> Double { d[0] * d[3] - d[1] * d[2] }
}

/// Log-pdf of a zero-mean multivariate normal at residual `y` (n×1) with
/// covariance `s` (n×n). Mirrors filterpy's `logpdf` used for IMM likelihoods.
func logpdf(_ y: Mat, _ s: Mat) -> Double {
    let n = y.rows
    let sInv = s.inverse()
    let tmp = sInv.matmul(y)
    var quad = 0.0
    for i in 0..<n { quad += y.d[i] * tmp.d[i] }
    let detS = (n == 2) ? s.det2x2() : det(s)
    let logDet = log(min(max(abs(detS), 1e-300), .greatestFiniteMagnitude))
    return -0.5 * (Double(n) * log(2 * Double.pi) + logDet + quad)
}

private func det(_ m: Mat) -> Double {
    let n = m.rows
    var a = m.d
    var det = 1.0
    for col in 0..<n {
        var piv = col
        var best = abs(a[col * n + col])
        for r in (col + 1)..<n {
            let v = abs(a[r * n + col])
            if v > best { best = v; piv = r }
        }
        if piv != col {
            for j in 0..<n { a.swapAt(col * n + j, piv * n + j) }
            det = -det
        }
        let pv = a[col * n + col]
        if abs(pv) < 1e-300 { return 0.0 }
        det *= pv
        for r in (col + 1)..<n {
            let f = a[r * n + col] / pv
            for j in col..<n { a[r * n + j] -= f * a[col * n + j] }
        }
    }
    return det
}
