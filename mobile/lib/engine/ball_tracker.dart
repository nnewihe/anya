import 'dart:collection';
import 'dart:math' as math;

import 'kalman.dart';
import 'linalg.dart';

/// A pixel-centre detection plus its YOLO confidence.
class Detection {
  final double x, y, conf;
  const Detection(this.x, this.y, this.conf);
}

typedef PerspectiveScale = double Function(double y);

double _noPerspective(double _) => 1.0;

/// Cheap perspective model needing only the analysis-frame height. Multiplier
/// in (farFloor, 1.0]: ~1.0 near the bottom, shrinking toward farFloor at top.
PerspectiveScale makeImageRowPerspective(double frameHeight,
    {double farFloor = 0.35}) {
  final h = math.max(1.0, frameHeight);
  return (double y) => math.max(farFloor, math.min(1.0, y / h));
}

/// Per-frame answer handed back to the rally state machine.
class TrackStatus {
  final bool hasMovingTrace;
  final String state;
  final List<double>? position; // [x, y] or null
  final double speedPxS;
  final double timeSinceDetection;
  final bool coasting;
  final int ballCount;
  final double maneuverProb;
  final double racketProb;
  final double bounceProb;
  final List<List<double>> trace;

  const TrackStatus({
    required this.hasMovingTrace,
    required this.state,
    required this.position,
    required this.speedPxS,
    required this.timeSinceDetection,
    required this.coasting,
    required this.ballCount,
    required this.maneuverProb,
    required this.racketProb,
    required this.bounceProb,
    required this.trace,
  });
}

class _ConfirmedTrack {
  final double fps;
  final double dt;
  final double motionWindowS;
  final double corroborationWindowS;
  final PerspectiveScale persp;

  late IMMEstimator imm;
  late List<Mat> baseQ;
  double lastDetectionT;
  int hits = 1;
  List<double> lastMeasuredPos;
  final Queue<List<double>> history = Queue(); // (t, x, y)
  final Queue<double> detTimes = Queue();

  _ConfirmedTrack(
    this.fps,
    double x,
    double y,
    double vx,
    double vy,
    double t,
    this.motionWindowS,
    this.corroborationWindowS, {
    double qSmooth = 5.0,
    double qRacket = 300.0,
    double qPos = 1.0,
    double qBounceVx = 20.0,
    double qBounceVy = 300.0,
    List<double>? muInit,
    List<List<double>>? m,
    PerspectiveScale? perspectiveScale,
  })  : dt = 1.0 / math.max(fps, 1e-6),
        persp = perspectiveScale ?? _noPerspective,
        lastDetectionT = t,
        lastMeasuredPos = [x, y] {
    final f = Mat.from([
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);
    final h = Mat.from([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ]);
    final r = Mat.identity(2).scaled(10.0);
    final p0 = Mat.identity(4).scaled(100.0);
    final x0 = Mat.colVec([x, y, vx, vy]);

    final qRacketPos = qRacket / 10.0;

    KalmanFilter mk(Mat q) => KalmanFilter(
        F: f.clone(),
        H: h.clone(),
        R: r.clone(),
        Q: q,
        P: p0.clone(),
        x: x0.clone());

    // Model 0 — smooth in-flight CV.
    final kf0 = mk(Mat.diag([qPos, qPos, qSmooth, qSmooth]));
    // Model 1 — racket impact: isotropic high-Q on position AND velocity.
    final kf1 = mk(Mat.diag([qRacketPos, qRacketPos, qRacket, qRacket]));
    // Model 2 — court bounce: anisotropic Q.
    final kf2 = mk(Mat.diag([qPos, qRacketPos, qBounceVx, qBounceVy]));

    final mu = muInit ?? [0.90, 0.05, 0.05];
    final trans = m ??
        [
          [0.92, 0.04, 0.04],
          [0.70, 0.25, 0.05],
          [0.70, 0.05, 0.25],
        ];
    imm = IMMEstimator([kf0, kf1, kf2], mu, trans);
    baseQ = [kf0.Q.clone(), kf1.Q.clone(), kf2.Q.clone()];

    history.add([t, x, y]);
    detTimes.add(t);
  }

  void predict() {
    final scale = persp(yPos);
    for (var i = 0; i < imm.filters.length; i++) {
      imm.filters[i].Q = baseQ[i].scaled(scale);
    }
    imm.predict();
  }

  void update(double x, double y, double t) {
    imm.update(Mat.colVec([x, y]));
    lastDetectionT = t;
    lastMeasuredPos = [x, y];
    hits += 1;
    detTimes.add(t);
  }

  List<double> get position => [imm.x.d[0], imm.x.d[1]];
  double get yPos => imm.x.d[1];
  double speedPxS() => math.sqrt(imm.x.d[2] * imm.x.d[2] + imm.x.d[3] * imm.x.d[3]);
  double positionUncertainty() =>
      imm.P.at(0, 0) + imm.P.at(1, 1); // trace of P[:2,:2]
  double get maneuverProb => 1.0 - imm.mu[0];
  double get racketProb => imm.mu[1];
  double get bounceProb => imm.mu[2];

  void record(double t, double now) {
    final p = position;
    history.add([t, p[0], p[1]]);
    final cutoff = now - motionWindowS;
    while (history.isNotEmpty && history.first[0] < cutoff) {
      history.removeFirst();
    }
    final detCutoff = now - corroborationWindowS;
    while (detTimes.isNotEmpty && detTimes.first < detCutoff) {
      detTimes.removeFirst();
    }
  }

  int recentDetCount() => detTimes.length;

  double recentSpanPx() {
    if (history.length < 2) return 0.0;
    final last = history.last;
    var maxD = 0.0;
    for (final h in history) {
      final d = math.sqrt(
          (h[1] - last[1]) * (h[1] - last[1]) + (h[2] - last[2]) * (h[2] - last[2]));
      if (d > maxD) maxD = d;
    }
    return maxD;
  }

  List<List<double>> trace() => [for (final h in history) [h[1], h[2]]];
  List<List<double>> traceWithTime() => [for (final h in history) [h[0], h[1], h[2]]];
}

class _Tentative {
  final List<List<double>> points = []; // (t, x, y)
  double lastT;

  _Tentative(double x, double y, double t) : lastT = t {
    points.add([t, x, y]);
  }

  void add(double x, double y, double t) {
    points.add([t, x, y]);
    lastT = t;
  }

  List<double> get lastXy => [points.last[1], points.last[2]];

  List<double> expectedNext(double t) {
    final lx = points.last[1], ly = points.last[2];
    if (points.length < 2) return [lx, ly];
    final p0 = points[points.length - 2];
    final p1 = points[points.length - 1];
    final segDt = p1[0] - p0[0];
    if (segDt <= 0) return [lx, ly];
    final vx = (p1[1] - p0[1]) / segDt, vy = (p1[2] - p0[2]) / segDt;
    final dt = t - p1[0];
    return [lx + vx * dt, ly + vy * dt];
  }

  double spanPx() {
    if (points.length < 2) return 0.0;
    final last = points.last;
    var maxD = 0.0;
    for (final p in points) {
      final d = math.sqrt((p[1] - last[1]) * (p[1] - last[1]) +
          (p[2] - last[2]) * (p[2] - last[2]));
      if (d > maxD) maxD = d;
    }
    return maxD;
  }

  List<double> velocity() {
    if (points.length < 2) return [0.0, 0.0];
    final p0 = points.first, p1 = points.last;
    final dt = p1[0] - p0[0];
    if (dt <= 0) return [0.0, 0.0];
    return [(p1[1] - p0[1]) / dt, (p1[2] - p0[2]) / dt];
  }
}

/// Maintains a single confirmed ball trajectory and reports whether a moving
/// trace is currently alive. Port of BallTrackManager.
class BallTrackManager {
  final double fps;
  final double dt;
  final PerspectiveScale persp;
  final double gateBasePx;
  final double gateUncertaintyK;
  final double seedGatePx;
  final double seedCoherencePx;
  final int confirmHits;
  final double confirmWindowS;
  final double missTimeoutS;
  final double motionWindowS;
  final double moveThreshPx;
  final int minRecentDets;
  final double corroborationWindowS;
  final double hijackAfterS;
  final double qSmooth;
  final double qManeuver;
  final double qPos;
  final double qBounceVx;
  final double qBounceVy;
  final double fallbackGateK;
  final double coastGateK;
  final double coastGateCapPx;

  // ignore: library_private_types_in_public_api
  _ConfirmedTrack? track; // internal state (exposed only within this library)
  // ignore: library_private_types_in_public_api
  List<_Tentative> tentatives = [];
  double? lastDetectionTime;
  List<List<double>> _lastTrace = [];

  BallTrackManager(
    this.fps, {
    PerspectiveScale? perspectiveScale,
    this.gateBasePx = 50.0,
    this.gateUncertaintyK = 0.6,
    this.seedGatePx = 100.0,
    this.seedCoherencePx = 38.0,
    this.confirmHits = 3,
    this.confirmWindowS = 0.6,
    this.missTimeoutS = 2.0,
    this.motionWindowS = 0.5,
    this.moveThreshPx = 30.0,
    this.minRecentDets = 3,
    this.corroborationWindowS = 2.0,
    this.hijackAfterS = 0.15,
    this.qSmooth = 5.0,
    this.qManeuver = 300.0,
    this.qPos = 1.0,
    this.qBounceVx = 20.0,
    this.qBounceVy = 300.0,
    this.fallbackGateK = 1.8,
    this.coastGateK = 0.5,
    this.coastGateCapPx = 400.0,
  })  : dt = 1.0 / math.max(fps, 1e-6),
        persp = perspectiveScale ?? _noPerspective;

  void reset() {
    track = null;
    tentatives = [];
    lastDetectionTime = null;
    _lastTrace = [];
  }

  TrackStatus update(List<Detection> detections, double now) {
    // 1. Predict the confirmed track forward.
    track?.predict();

    // 2. Associate one detection to the confirmed track.
    final used = List<bool>.filled(detections.length, false);
    final t = track;
    if (t != null && detections.isNotEmpty) {
      final pos = t.position;
      final tx = pos[0], ty = pos[1];
      final scale = persp(t.yPos);
      var gate = gateBasePx * scale +
          gateUncertaintyK * math.sqrt(math.max(t.positionUncertainty(), 0.0));
      final tsd = now - t.lastDetectionT;
      gate += math.min(coastGateK * t.speedPxS() * tsd, coastGateCapPx);
      var bestI = -1;
      var bestD = gate;
      for (var i = 0; i < detections.length; i++) {
        final d = math.sqrt((detections[i].x - tx) * (detections[i].x - tx) +
            (detections[i].y - ty) * (detections[i].y - ty));
        if (d <= bestD) {
          bestD = d;
          bestI = i;
        }
      }

      // Fallback gate around the last measured position.
      if (bestI < 0) {
        final lmx = t.lastMeasuredPos[0], lmy = t.lastMeasuredPos[1];
        final fbGate = gateBasePx * fallbackGateK * scale;
        var fbBestI = -1;
        var fbBestD = fbGate;
        for (var i = 0; i < detections.length; i++) {
          final d = math.sqrt((detections[i].x - lmx) * (detections[i].x - lmx) +
              (detections[i].y - lmy) * (detections[i].y - lmy));
          if (d <= fbBestD) {
            fbBestD = d;
            fbBestI = i;
          }
        }
        bestI = fbBestI;
      }

      if (bestI >= 0) {
        t.update(detections[bestI].x, detections[bestI].y, now);
        used[bestI] = true;
        lastDetectionTime = now;
      }
    }

    // 3. Feed leftovers to tentative seeds; try to promote one.
    for (var i = 0; i < detections.length; i++) {
      if (!used[i]) _feedTentative(detections[i].x, detections[i].y, now);
    }
    _pruneTentatives(now);
    _tryPromote(now);

    // 4. Record position into the motion window.
    track?.record(now, now);

    final status = _status(detections.length, now);
    if (status.state == 'lost') track = null;
    return status;
  }

  void _feedTentative(double x, double y, double now) {
    final scale = persp(y);
    _Tentative? best;
    var bestD = double.infinity;
    for (final tnt in tentatives) {
      final e = tnt.expectedNext(now);
      final gate = (tnt.points.length < 2 ? seedGatePx : seedCoherencePx) * scale;
      final d = math.sqrt((x - e[0]) * (x - e[0]) + (y - e[1]) * (y - e[1]));
      if (d <= gate && d < bestD) {
        bestD = d;
        best = tnt;
      }
    }
    if (best != null) {
      best.add(x, y, now);
    } else {
      tentatives.add(_Tentative(x, y, now));
    }
  }

  void _pruneTentatives(double now) {
    tentatives =
        tentatives.where((t) => now - t.points.first[0] <= confirmWindowS).toList();
  }

  void _tryPromote(double now) {
    final t = track;
    if (t != null) {
      if (now - t.lastDetectionT <= hijackAfterS) return;
    }
    final promotable = tentatives
        .where((s) =>
            s.points.length >= confirmHits &&
            s.spanPx() > moveThreshPx * persp(s.lastXy[1]))
        .toList();
    if (promotable.isEmpty) return;
    promotable.sort((a, b) {
      if (a.points.length != b.points.length) {
        return a.points.length.compareTo(b.points.length);
      }
      return a.spanPx().compareTo(b.spanPx());
    });
    final best = promotable.last; // max (len, span)
    final xy = best.lastXy;
    final v = best.velocity();
    final tLast = best.points.last[0];
    final nt = _ConfirmedTrack(
      fps, xy[0], xy[1], v[0], v[1], tLast, motionWindowS, corroborationWindowS,
      qSmooth: qSmooth,
      qRacket: qManeuver,
      qPos: qPos,
      qBounceVx: qBounceVx,
      qBounceVy: qBounceVy,
      perspectiveScale: persp,
    );
    // Backfill motion window + corroboration from the seed.
    nt.history.clear();
    nt.detTimes.clear();
    for (final p in best.points) {
      nt.history.add([p[0], p[1], p[2]]);
      nt.detTimes.add(p[0]);
    }
    track = nt;
    lastDetectionTime = tLast;
    tentatives.remove(best);
  }

  TrackStatus _status(int ballCount, double now) {
    final t = track;
    if (t == null) {
      _lastTrace = [];
      return TrackStatus(
        hasMovingTrace: false,
        state: 'none',
        position: null,
        speedPxS: 0.0,
        timeSinceDetection: 0.0,
        coasting: false,
        ballCount: ballCount,
        maneuverProb: 0.0,
        racketProb: 0.0,
        bounceProb: 0.0,
        trace: const [],
      );
    }

    final tsd = now - t.lastDetectionT;
    final lost = tsd > missTimeoutS;
    final coasting = (!lost) && tsd > (1.5 * dt);
    final moving = t.recentSpanPx() > moveThreshPx * persp(t.yPos);
    final corroborated = t.recentDetCount() >= minRecentDets;

    _lastTrace = t.traceWithTime();

    String state;
    bool alive;
    if (lost) {
      state = 'lost';
      alive = false;
    } else if (!corroborated) {
      state = 'fading';
      alive = false;
    } else if (!moving) {
      state = 'stopped';
      alive = false;
    } else {
      state = coasting ? 'coasting' : 'moving';
      alive = true;
    }

    return TrackStatus(
      hasMovingTrace: alive,
      state: state,
      position: t.position,
      speedPxS: t.speedPxS(),
      timeSinceDetection: tsd,
      coasting: coasting,
      ballCount: ballCount,
      maneuverProb: t.maneuverProb,
      racketProb: t.racketProb,
      bounceProb: t.bounceProb,
      trace: t.trace(),
    );
  }

  List<List<double>> tracePoints() => List.of(_lastTrace);
}
