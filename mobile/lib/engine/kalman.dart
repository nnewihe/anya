import 'dart:math' as math;

import 'linalg.dart';

/// Port of filterpy's `KalmanFilter` for the fixed shape used by the ball
/// tracker (dim_x=4, dim_z=2). Same predict/update equations, and the same
/// multivariate-normal `likelihood` used by the IMM.
class KalmanFilter {
  Mat F, H, R, Q, P, x;
  late Mat y; // residual
  late Mat S; // system uncertainty
  double logLikelihood = 0.0;
  double likelihood = 1.0;

  KalmanFilter({
    required this.F,
    required this.H,
    required this.R,
    required this.Q,
    required this.P,
    required this.x,
  });

  void predict() {
    x = F.matmul(x);
    P = F.matmul(P).matmul(F.transpose()) + Q;
  }

  void update(Mat z) {
    y = z - H.matmul(x); // residual
    final ht = H.transpose();
    final pht = P.matmul(ht);
    S = H.matmul(pht) + R;
    final k = pht.matmul(S.inverse());
    x = x + k.matmul(y);
    final iKh = Mat.identity(x.rows) - k.matmul(H);
    P = iKh.matmul(P).matmul(iKh.transpose()) +
        k.matmul(R).matmul(k.transpose());
    logLikelihood = logpdf(y, S);
    likelihood = math.exp(logLikelihood);
    if (likelihood == 0.0) likelihood = 5e-324; // smallest positive double
  }
}

/// Port of filterpy's `IMMEstimator`. Interacting-multiple-model mixing over a
/// set of Kalman filters with mode probabilities `mu` and Markov transition
/// matrix `M`.
class IMMEstimator {
  final List<KalmanFilter> filters;
  final int n;
  List<double> mu;
  final List<List<double>> M;
  late List<double> cbar;
  late List<List<double>> omega;
  late Mat x;
  late Mat P;

  IMMEstimator(this.filters, List<double> muInit, this.M)
      : n = filters.length,
        mu = _normalize(muInit) {
    omega = [for (var i = 0; i < n; i++) List<double>.filled(n, 0.0)];
    _computeMixingProbabilities();
    _computeStateEstimate();
  }

  static List<double> _normalize(List<double> v) {
    final s = v.fold<double>(0.0, (a, b) => a + b);
    return [for (final e in v) e / s];
  }

  void predict() {
    final dim = filters[0].x.rows;
    final xs = <Mat>[];
    final ps = <Mat>[];
    // Mixed initial conditions for each target filter i using column omega[:, i].
    for (var i = 0; i < n; i++) {
      final xMix = Mat(dim, 1);
      for (var j = 0; j < n; j++) {
        final w = omega[j][i];
        for (var r = 0; r < dim; r++) {
          xMix.d[r] += filters[j].x.d[r] * w;
        }
      }
      xs.add(xMix);
    }
    for (var i = 0; i < n; i++) {
      var pMix = Mat(dim, dim);
      for (var j = 0; j < n; j++) {
        final w = omega[j][i];
        final yv = filters[j].x - xs[i];
        pMix = pMix + (yv.outer(yv) + filters[j].P).scaled(w);
      }
      ps.add(pMix);
    }
    for (var i = 0; i < n; i++) {
      filters[i].x = xs[i].clone();
      filters[i].P = ps[i].clone();
      filters[i].predict();
    }
    _computeStateEstimate();
  }

  void update(Mat z) {
    final likelihood = List<double>.filled(n, 0.0);
    for (var i = 0; i < n; i++) {
      filters[i].update(z);
      likelihood[i] = filters[i].likelihood;
    }
    // mu = cbar * likelihood, normalized
    var sum = 0.0;
    final newMu = List<double>.filled(n, 0.0);
    for (var i = 0; i < n; i++) {
      newMu[i] = cbar[i] * likelihood[i];
      sum += newMu[i];
    }
    for (var i = 0; i < n; i++) {
      newMu[i] /= sum;
    }
    mu = newMu;
    _computeMixingProbabilities();
    _computeStateEstimate();
  }

  void _computeMixingProbabilities() {
    // cbar[j] = sum_i mu[i] * M[i][j]
    cbar = List<double>.filled(n, 0.0);
    for (var j = 0; j < n; j++) {
      var s = 0.0;
      for (var i = 0; i < n; i++) {
        s += mu[i] * M[i][j];
      }
      cbar[j] = s;
    }
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        omega[i][j] = (M[i][j] * mu[i]) / cbar[j];
      }
    }
  }

  void _computeStateEstimate() {
    final dim = filters[0].x.rows;
    x = Mat(dim, 1);
    for (var i = 0; i < n; i++) {
      for (var r = 0; r < dim; r++) {
        x.d[r] += filters[i].x.d[r] * mu[i];
      }
    }
    P = Mat(dim, dim);
    for (var i = 0; i < n; i++) {
      final yv = filters[i].x - x;
      P = P + (yv.outer(yv) + filters[i].P).scaled(mu[i]);
    }
  }
}
