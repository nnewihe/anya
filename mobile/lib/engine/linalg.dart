import 'dart:math' as math;

/// Minimal dense matrix for the Kalman / IMM port. Row-major `Float64List`
/// stored as `List<List<double>>` for readability at these tiny sizes (≤4×4).
///
/// Only the operations the ball tracker needs are implemented, matching
/// numpy/filterpy semantics: matmul, transpose, add/sub, scale, general
/// inverse (Gauss–Jordan), and the multivariate-normal log-pdf used for the
/// IMM likelihoods.
class Mat {
  final int rows;
  final int cols;
  final List<double> d; // row-major, length rows*cols

  Mat(this.rows, this.cols) : d = List<double>.filled(rows * cols, 0.0);

  Mat.from(List<List<double>> r)
      : rows = r.length,
        cols = r[0].length,
        d = [for (final row in r) ...row];

  factory Mat.identity(int n) {
    final m = Mat(n, n);
    for (var i = 0; i < n; i++) {
      m.d[i * n + i] = 1.0;
    }
    return m;
  }

  factory Mat.colVec(List<double> v) {
    final m = Mat(v.length, 1);
    for (var i = 0; i < v.length; i++) {
      m.d[i] = v[i];
    }
    return m;
  }

  factory Mat.diag(List<double> v) {
    final n = v.length;
    final m = Mat(n, n);
    for (var i = 0; i < n; i++) {
      m.d[i * n + i] = v[i];
    }
    return m;
  }

  double at(int r, int c) => d[r * cols + c];
  void set(int r, int c, double v) => d[r * cols + c] = v;

  Mat clone() {
    final m = Mat(rows, cols);
    for (var i = 0; i < d.length; i++) {
      m.d[i] = d[i];
    }
    return m;
  }

  Mat matmul(Mat b) {
    assert(cols == b.rows);
    final out = Mat(rows, b.cols);
    for (var i = 0; i < rows; i++) {
      for (var k = 0; k < cols; k++) {
        final a = d[i * cols + k];
        if (a == 0.0) continue;
        for (var j = 0; j < b.cols; j++) {
          out.d[i * b.cols + j] += a * b.d[k * b.cols + j];
        }
      }
    }
    return out;
  }

  Mat transpose() {
    final out = Mat(cols, rows);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        out.d[j * rows + i] = d[i * cols + j];
      }
    }
    return out;
  }

  Mat operator +(Mat b) {
    final out = Mat(rows, cols);
    for (var i = 0; i < d.length; i++) {
      out.d[i] = d[i] + b.d[i];
    }
    return out;
  }

  Mat operator -(Mat b) {
    final out = Mat(rows, cols);
    for (var i = 0; i < d.length; i++) {
      out.d[i] = d[i] - b.d[i];
    }
    return out;
  }

  Mat scaled(double s) {
    final out = Mat(rows, cols);
    for (var i = 0; i < d.length; i++) {
      out.d[i] = d[i] * s;
    }
    return out;
  }

  /// Outer product of two column vectors (this·otherᵀ), both n×1.
  Mat outer(Mat other) {
    final out = Mat(rows, other.rows);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < other.rows; j++) {
        out.d[i * other.rows + j] = d[i] * other.d[j];
      }
    }
    return out;
  }

  /// General square-matrix inverse via Gauss–Jordan with partial pivoting.
  Mat inverse() {
    assert(rows == cols);
    final n = rows;
    // Augmented [A | I]
    final a = List<double>.filled(n * 2 * n, 0.0);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        a[i * 2 * n + j] = d[i * n + j];
      }
      a[i * 2 * n + n + i] = 1.0;
    }
    for (var col = 0; col < n; col++) {
      // pivot
      var piv = col;
      var best = a[col * 2 * n + col].abs();
      for (var r = col + 1; r < n; r++) {
        final v = a[r * 2 * n + col].abs();
        if (v > best) {
          best = v;
          piv = r;
        }
      }
      if (piv != col) {
        for (var j = 0; j < 2 * n; j++) {
          final t = a[col * 2 * n + j];
          a[col * 2 * n + j] = a[piv * 2 * n + j];
          a[piv * 2 * n + j] = t;
        }
      }
      final pivVal = a[col * 2 * n + col];
      if (pivVal.abs() < 1e-12) {
        continue; // singular-ish; leave row (mirrors numpy blowing up rarely)
      }
      final inv = 1.0 / pivVal;
      for (var j = 0; j < 2 * n; j++) {
        a[col * 2 * n + j] *= inv;
      }
      for (var r = 0; r < n; r++) {
        if (r == col) continue;
        final f = a[r * 2 * n + col];
        if (f == 0.0) continue;
        for (var j = 0; j < 2 * n; j++) {
          a[r * 2 * n + j] -= f * a[col * 2 * n + j];
        }
      }
    }
    final out = Mat(n, n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        out.d[i * n + j] = a[i * 2 * n + n + j];
      }
    }
    return out;
  }

  double det2x2() => d[0] * d[3] - d[1] * d[2];
}

/// Log-pdf of a zero-mean multivariate normal at residual `y` (n×1) with
/// covariance `s` (n×n). Mirrors filterpy's `logpdf` used for IMM likelihoods.
double logpdf(Mat y, Mat s) {
  final n = y.rows;
  final sInv = s.inverse();
  // quad = yᵀ S⁻¹ y
  final tmp = sInv.matmul(y); // n×1
  var quad = 0.0;
  for (var i = 0; i < n; i++) {
    quad += y.d[i] * tmp.d[i];
  }
  final detS = (n == 2) ? s.det2x2() : _det(s);
  final logDet = math.log(detS.abs().clamp(1e-300, double.infinity));
  return -0.5 * (n * math.log(2 * math.pi) + logDet + quad);
}

double _det(Mat m) {
  // LU-free small determinant via Gaussian elimination (n≤4 here).
  final n = m.rows;
  final a = [for (var i = 0; i < n * n; i++) m.d[i]];
  var det = 1.0;
  for (var col = 0; col < n; col++) {
    var piv = col;
    var best = a[col * n + col].abs();
    for (var r = col + 1; r < n; r++) {
      final v = a[r * n + col].abs();
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (piv != col) {
      for (var j = 0; j < n; j++) {
        final t = a[col * n + j];
        a[col * n + j] = a[piv * n + j];
        a[piv * n + j] = t;
      }
      det = -det;
    }
    final pv = a[col * n + col];
    if (pv.abs() < 1e-300) return 0.0;
    det *= pv;
    for (var r = col + 1; r < n; r++) {
      final f = a[r * n + col] / pv;
      for (var j = col; j < n; j++) {
        a[r * n + j] -= f * a[col * n + j];
      }
    }
  }
  return det;
}
