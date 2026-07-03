import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import 'ball_tracker.dart';
import 'inference.dart';
import 'telemetry.dart';

/// A detected rally segment in source-video time.
class RallySegment {
  final double start;
  final double end;
  final String origin; // "near" | "far"
  const RallySegment(this.start, this.end, this.origin);
}

// Tunables — ported verbatim from rally_detector.py.
const double rallyGapThresholdSec = 4.0;
const double rallyPreRollSec = 1.5;
const double rallyEndPadSec = 1.0;

const double couplingWindowSec = 0.40;
const double couplingMinPlayerSpeed = 25.0;
const double couplingRatioMax = 0.50;

const double hmmPStay = 0.9355;
const double hmmPCorrect = 0.85;

const double racketSpikeThresh = 0.25;
const int minRacketFrames = 5;
const double minSegmentSec = 2.5;

/// Sliding-window velocity estimator (px/s) over (t,x,y) samples.
class SmoothedVelocity {
  final double windowSec;
  final Queue<List<double>> _pts = Queue();
  SmoothedVelocity(this.windowSec);

  void add(double t, double x, double y) {
    _pts.add([t, x, y]);
    final cutoff = t - windowSec;
    while (_pts.isNotEmpty && _pts.first[0] < cutoff) {
      _pts.removeFirst();
    }
  }

  List<double>? velocity() {
    if (_pts.length < 2) return null;
    final a = _pts.first, b = _pts.last;
    final dt = b[0] - a[0];
    if (dt <= 0) return null;
    return [(b[1] - a[1]) / dt, (b[2] - a[2]) / dt];
  }
}

({double? ratio, double playerSpeed}) _couplingRatio(
    List<double>? vBall, List<double>? vPlayer) {
  if (vBall == null || vPlayer == null) return (ratio: null, playerSpeed: 0.0);
  final ballSpeed = math.sqrt(vBall[0] * vBall[0] + vBall[1] * vBall[1]);
  final playerSpeed =
      math.sqrt(vPlayer[0] * vPlayer[0] + vPlayer[1] * vPlayer[1]);
  if (ballSpeed < 1e-6) return (ratio: null, playerSpeed: playerSpeed);
  final dx = vBall[0] - vPlayer[0], dy = vBall[1] - vPlayer[1];
  return (ratio: math.sqrt(dx * dx + dy * dy) / ballSpeed, playerSpeed: playerSpeed);
}

bool _isCarried(List<double>? vBall, List<double>? vPlayer) {
  final r = _couplingRatio(vBall, vPlayer);
  if (r.ratio == null) return false;
  return r.ratio! < couplingRatioMax && r.playerSpeed >= couplingMinPlayerSpeed;
}

List<double>? _boxCenter(List<double>? box) =>
    box == null ? null : [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0];

String _originSide(List<double>? ballXy, List<double>? nearBox, List<double>? farBox) {
  if (ballXy == null) return 'near';
  final nearC = _boxCenter(nearBox);
  final farC = _boxCenter(farBox);
  if (nearC == null && farC == null) return 'near';
  if (farC == null) return 'near';
  if (nearC == null) return 'far';
  final dNear = math.sqrt(math.pow(ballXy[0] - nearC[0], 2).toDouble() +
      math.pow(ballXy[1] - nearC[1], 2).toDouble());
  final dFar = math.sqrt(math.pow(ballXy[0] - farC[0], 2).toDouble() +
      math.pow(ballXy[1] - farC[1], 2).toDouble());
  return dFar < dNear ? 'far' : 'near';
}

class _Strength {
  int racketFrames;
  double duration;
  _Strength(this.racketFrames, this.duration);
}

class _RawSeg {
  double start;
  double end;
  String origin;
  _Strength strength;
  _RawSeg(this.start, this.end, this.origin, this.strength);
}

bool _segmentIsStrong(_Strength s) =>
    s.racketFrames >= minRacketFrames || s.duration >= minSegmentSec;

/// Viterbi decoding of the sticky serving-side HMM. States/obs: near=0, far=1.
List<String> viterbi(List<String> obsSides,
    {double pStay = hmmPStay, double pCorrect = hmmPCorrect}) {
  final n = obsSides.length;
  if (n == 0) return [];
  if (n == 1) return List.of(obsSides);
  const sides = ['near', 'far'];
  int idx(String s) => s == 'near' ? 0 : 1;

  final logTrans = [
    [math.log(pStay), math.log(1.0 - pStay)],
    [math.log(1.0 - pStay), math.log(pStay)],
  ];
  final logEmit = [
    [math.log(pCorrect), math.log(1.0 - pCorrect)],
    [math.log(1.0 - pCorrect), math.log(pCorrect)],
  ];
  final logInit = [math.log(0.5), math.log(0.5)];
  final obs = [for (final o in obsSides) idx(o)];

  final delta = [
    logInit[0] + logEmit[0][obs[0]],
    logInit[1] + logEmit[1][obs[0]],
  ];
  final psi = [for (var i = 0; i < n; i++) List<int>.filled(2, 0)];

  for (var t = 1; t < n; t++) {
    final newDelta = List<double>.filled(2, 0.0);
    for (var s = 0; s < 2; s++) {
      var best = 0;
      var bestScore = delta[0] + logTrans[0][s];
      final s1 = delta[1] + logTrans[1][s];
      if (s1 > bestScore) {
        bestScore = s1;
        best = 1;
      }
      newDelta[s] = bestScore + logEmit[s][obs[t]];
      psi[t][s] = best;
    }
    delta[0] = newDelta[0];
    delta[1] = newDelta[1];
  }

  final path = List<int>.filled(n, 0);
  path[n - 1] = delta[1] > delta[0] ? 1 : 0;
  for (var t = n - 2; t >= 0; t--) {
    path[t] = psi[t + 1][path[t + 1]];
  }
  return [for (final s in path) sides[s]];
}

List<RallySegment> _hmmFilter(List<_RawSeg> segments) {
  if (segments.isEmpty) return [];
  final decoded = viterbi([for (final s in segments) s.origin]);
  final kept = <RallySegment>[];
  for (var i = 0; i < segments.length; i++) {
    final seg = segments[i];
    if (seg.origin == decoded[i]) {
      kept.add(RallySegment(seg.start, seg.end, seg.origin));
    } else if (_segmentIsStrong(seg.strength)) {
      kept.add(RallySegment(seg.start, seg.end, decoded[i]));
    }
    // weak + disagreeing → dropped
  }
  return kept;
}

List<_RawSeg> _mergeSegments(List<_RawSeg> segments,
    {double gapThresholdSec = rallyGapThresholdSec}) {
  if (segments.isEmpty) return [];
  final merged = <_RawSeg>[
    _RawSeg(segments[0].start, segments[0].end, segments[0].origin,
        _Strength(segments[0].strength.racketFrames, segments[0].strength.duration))
  ];
  for (var i = 1; i < segments.length; i++) {
    final s = segments[i];
    final last = merged.last;
    if (s.start - last.end < gapThresholdSec) {
      last.end = s.end;
      last.strength.racketFrames += s.strength.racketFrames;
      last.strength.duration += s.strength.duration;
    } else {
      merged.add(_RawSeg(s.start, s.end, s.origin,
          _Strength(s.strength.racketFrames, s.strength.duration)));
    }
  }
  return merged;
}

List<RallySegment> _applyPreRoll(List<RallySegment> segments,
    {double preRollSec = rallyPreRollSec}) {
  return [
    for (final s in segments)
      RallySegment(math.max(0.0, s.start - preRollSec), s.end, s.origin)
  ];
}

/// Trace-driven rally detector — port of collect_rally_segments. Consumes an
/// ordered stream of rgb24 frames and returns the final segment list.
Future<List<RallySegment>> collectRallySegments({
  required Stream<Uint8List> frames,
  required double fps,
  required int totalFrames,
  required TelemetryProvider telemetry,
  required BallTrackManager ballTracker,
  void Function(int current, int total)? progressCb,
}) async {
  final videoDurationSec =
      totalFrames > 0 ? totalFrames / fps : double.infinity;

  final ballVel = SmoothedVelocity(couplingWindowSec);
  final playerVel = SmoothedVelocity(couplingWindowSec);

  final rawSegments = <_RawSeg>[];
  double? segStart;
  var segOrigin = 'near';
  var segRacketFrames = 0;
  var lastTs = 0.0;
  var frameNum = 0;

  await for (final rgb in frames) {
    frameNum += 1;
    final tensor = letterboxRgbToTensor(rgb);
    final tel = telemetry.processFrame(tensor);
    lastTs = tel.timestamp;

    if (progressCb != null && frameNum % 30 == 0) {
      progressCb(frameNum, totalFrames);
    }

    final nearBox = tel.nearPlayerBox;
    final farBox = tel.farPlayerBox;
    final status = ballTracker.update(tel.activeBallCandidates, tel.timestamp);

    if (status.position != null) {
      ballVel.add(tel.timestamp, status.position![0], status.position![1]);
    }
    if (nearBox != null) {
      // Feet point (bottom-centre), not box centroid: carry-coupling compares
      // the ball's pixel velocity against the player's WALKING motion, and
      // feet track that ground motion directly (the centroid also bobs with
      // torso/racket movement).
      playerVel.add(
          tel.timestamp, (nearBox[0] + nearBox[2]) / 2.0, nearBox[3]);
    }

    var carried = false;
    if (status.hasMovingTrace && status.position != null) {
      carried = _isCarried(ballVel.velocity(), playerVel.velocity());
    }
    final traceActive = status.hasMovingTrace && !carried;

    if (segStart != null && traceActive) {
      if (status.racketProb > racketSpikeThresh) segRacketFrames += 1;
    }

    if (traceActive) {
      if (segStart == null) {
        segStart = tel.timestamp;
        segOrigin = _originSide(status.position, nearBox, farBox);
        segRacketFrames = 0;
      }
    } else {
      if (segStart != null) {
        final rawEnd = ballTracker.lastDetectionTime ?? tel.timestamp;
        final paddedEnd = math.min(rawEnd + rallyEndPadSec, videoDurationSec);
        rawSegments.add(_RawSeg(segStart, paddedEnd, segOrigin,
            _Strength(segRacketFrames, rawEnd - segStart)));
        segStart = null;
      }
    }
  }

  // Flush a still-open segment at end of stream.
  if (segStart != null) {
    final rawEnd = ballTracker.lastDetectionTime ?? lastTs;
    final paddedEnd = math.min(rawEnd + rallyEndPadSec, videoDurationSec);
    rawSegments.add(_RawSeg(segStart, paddedEnd, segOrigin,
        _Strength(segRacketFrames, rawEnd - segStart)));
  }

  final merged = _mergeSegments(rawSegments);
  final filtered = _hmmFilter(merged);
  return _applyPreRoll(filtered);
}
