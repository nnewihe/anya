import 'dart:convert';
import 'dart:math' as math;

import 'ball_tracker.dart'
    show Detection, BallTrackManager, makeImageRowPerspective;
import 'kalman.dart';
import 'linalg.dart';

/// Stage-2 of the dead-time cutter, ported from pipeline/point_segmenter.py.
///
/// Pure time-series logic over the stage-1 telemetry JSONL — no models, no
/// video, no ONNX — so it ports 1:1 to Dart and can be golden-mastered against
/// the Python segment_match output on the same telemetry.  Reuses the existing
/// Dart ball_tracker.dart (IMM replay) and kalman.dart (toss tracker).
///
/// This file is built in slices; see the Phase-3 task breakdown.  Slice 3a
/// (here): config, data model, JSONL loading, static-candidate suppression.

// ===========================================================================
// Configuration — every tunable knob (mirrors SegmenterConfig defaults).
// Mutable like the Python dataclass: court_length_ft / frame_height_px are
// overridden from the telemetry meta in segment_match.
// ===========================================================================
class SegmenterConfig {
  // Court constants (overridden from telemetry meta).
  double courtLengthFt = 78.0;
  double frameHeightPx = 540.0;

  // Ready-band gating (near side).
  List<double> nearBandFt = const [-0.5, 3.5];
  double readyDwellS = 0.2;
  double bandWindowS = 2.0;
  double bandOutRatio = 0.25;

  // Static-candidate suppression.
  int staticCellPx = 16;
  double staticFrac = 0.04;

  // Near-serve scoring: Kalman-filtered toss (see _TossTracker).
  double tossConfFloor = 0.5;
  int tossConfirmFrames = 2;
  double tossMinRiseDurationS = 0.0;
  double tossSeedGatePx = 60.0;
  double tossSeedMaxDtS = 0.25;
  double tossAssocGatePx = 45.0;
  double tossCoastMaxS = 0.20;
  double tossMinVyPxS = 80.0;
  double tossMaxHorizRatio = 1.2;
  double tossKfQPos = 4.0;
  double tossKfQVel = 800.0;
  double tossKfRPx = 16.0;
  double tossBandGraceS = 0.35;
  double trophyWeight = 0.2;
  double tossWeight = 0.8;
  double serveScoreThreshold = 0.55;
  double serveEventWindowS = 1.2;

  // Far-serve detection (ball-trace only).
  double farOriginPadPx = 45.0;
  int farFeetMinSamples = 300;
  double farTraceOriginFrac = 0.6;
  double farTraceHeadS = 2.5;
  double farTraceQuietS = 4.0;
  double farTraceMinIntervalS = 0.25;
  double farPresenceSlackS = 1.0;

  // Serve-event bookkeeping.
  double minServeSeparationS = 8.0;
  double serveRearmS = 2.0;

  // Serve-trace confirmation.
  double confirmWindowS = 4.0;
  double traceDownwardPxS = 40.0;
  double traceHorizontalPxS = 30.0;

  // Serving-side HMM.
  double hmmPStay = 0.9355;
  double hmmPCorrect = 0.85;

  // Ball-trace liveness during replay.
  double moveVelocityFloorPxS = 20.0;
  double aliveMergeGapS = 0.6;
  double racketSpikeThresh = 0.25;
  double inboxAcceptPx = 35.0;

  // Carried-ball suppression.
  double couplingWindowS = 0.40;
  double couplingMinPlayerSpeed = 25.0;
  double couplingRatioMax = 0.50;

  // Point-end chaining.
  double serveChainWindowS = 5.0;
  double chainGapS = 4.0;
  double chainGapActiveS = 8.0;
  double activityGapS = 2.5;
  double activityExtendMaxS = 12.0;
  double fallbackPointS = 6.0;
  double maxPointS = 60.0;
  double minPointS = 1.5;
  double nextServeGuardS = 1.5;

  // Player-kinematics rally cues (near player only).
  double speedWindowS = 0.4;
  double speedMinDtS = 0.15;
  double reversalSpeedFtS = 3.0;

  // Output segments.
  double preRollS = 2.0;
  double farPreRollS = 4.5;
  double endPadS = 2.0;

  SegmenterConfig();
}

// ===========================================================================
// Telemetry data model
// ===========================================================================
class FrameRecord {
  final int f;
  final double t;
  final List<int>? nearBox; // [x1,y1,x2,y2]
  final List<double>? nearWorld; // [wx, wy]
  final List<int>? farBox;
  final bool farHeld;
  final List<double>? farWorld;
  List<Detection> balls;
  List<Detection> toss;
  List<Detection> fballs;
  List<Detection> rballs;
  final double trophy;
  final double stgcn;

  FrameRecord({
    required this.f,
    required this.t,
    required this.nearBox,
    required this.nearWorld,
    required this.farBox,
    required this.farHeld,
    required this.farWorld,
    required this.balls,
    required this.toss,
    required this.fballs,
    required this.rballs,
    required this.trophy,
    required this.stgcn,
  });
}

class MatchTelemetry {
  final Map<String, dynamic> meta;
  final List<FrameRecord> records;
  final List<double> ts;
  final double fps;
  bool staticSuppressed = false;

  MatchTelemetry(this.meta, this.records)
      : ts = [for (final r in records) r.t],
        fps = (_numOr(meta['fps'], 30.0)) /
            _intOr(meta['stride'], 1).clamp(1, 1 << 30);

  double get duration => ts.isEmpty ? 0.0 : ts.last;

  /// Record indices covering [t0, t1) — (bisectLeft(t0), bisectLeft(t1)).
  List<int> indexRange(double t0, double t1) =>
      [_bisectLeft(ts, t0), _bisectLeft(ts, t1)];

  List<FrameRecord> slice(double t0, double t1) {
    final r = indexRange(t0, t1);
    return records.sublist(r[0], r[1]);
  }
}

List<Detection> _dets(dynamic v) {
  if (v == null) return <Detection>[];
  return [
    for (final b in v as List)
      Detection((b[0] as num).toDouble(), (b[1] as num).toDouble(),
          (b[2] as num).toDouble())
  ];
}

List<int>? _ints(dynamic v) =>
    v == null ? null : [for (final e in v as List) (e as num).round()];

List<double>? _doubles(dynamic v) =>
    v == null ? null : [for (final e in v as List) (e as num).toDouble()];

double _numOr(dynamic v, double d) => v == null ? d : (v as num).toDouble();
int _intOr(dynamic v, int d) => v == null ? d : (v as num).toInt();

/// Parse the stage-1 JSONL (meta line + record lines) — mirrors load_telemetry.
MatchTelemetry loadTelemetry(String jsonl) {
  Map<String, dynamic> meta = {};
  final records = <FrameRecord>[];
  for (final rawLine in const LineSplitter().convert(jsonl)) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;
    final obj = jsonDecode(line) as Map<String, dynamic>;
    if (obj.containsKey('meta')) {
      meta = obj['meta'] as Map<String, dynamic>;
      continue;
    }
    records.add(FrameRecord(
      f: (obj['f'] as num).toInt(),
      t: (obj['t'] as num).toDouble(),
      nearBox: _ints(obj['np']),
      nearWorld: _doubles(obj['npw']),
      farBox: _ints(obj['fp']),
      farHeld: (_intOr(obj['fph'], 0)) != 0,
      farWorld: _doubles(obj['fpw']),
      balls: _dets(obj['balls']),
      toss: _dets(obj['toss']),
      fballs: _dets(obj['fballs']),
      rballs: _dets(obj['rballs']),
      trophy: _numOr(obj['trophy'], 0.0),
      stgcn: _numOr(obj['stgcn'], 0.0),
    ));
  }
  return MatchTelemetry(meta, records);
}

int _bisectLeft(List<double> a, double x) {
  var lo = 0, hi = a.length;
  while (lo < hi) {
    final mid = (lo + hi) >> 1;
    if (a[mid] < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ===========================================================================
// Static-candidate suppression
// ===========================================================================

/// Remove toss/fballs/rballs candidates that sit in 'hot' cells — grid cells
/// where a candidate appears in more than staticFrac of all frames.  Real
/// balls move; a candidate that lives in one cell for minutes is a static
/// false positive that starves the toss tracker.  Mutates records in place
/// (idempotent per loaded telemetry); returns the number dropped.
int suppressStaticCandidates(MatchTelemetry match, SegmenterConfig cfg) {
  if (match.staticSuppressed) return 0;
  final n = match.records.length;
  if (n == 0) return 0;
  final cell = cfg.staticCellPx < 4 ? 4 : cfg.staticCellPx;
  final counts = <int, int>{};
  for (final rec in match.records) {
    final seen = <int>{};
    for (final list in [rec.toss, rec.fballs, rec.rballs]) {
      for (final d in list) {
        // pack (col, row) into one int key
        seen.add((d.x ~/ cell) * 100000 + (d.y ~/ cell));
      }
    }
    for (final key in seen) {
      counts[key] = (counts[key] ?? 0) + 1;
    }
  }
  final hot = <int>{
    for (final e in counts.entries)
      if (e.value / n > cfg.staticFrac) e.key
  };
  if (hot.isEmpty) {
    match.staticSuppressed = true;
    return 0;
  }
  bool isHot(Detection d) =>
      hot.contains((d.x ~/ cell) * 100000 + (d.y ~/ cell));
  List<Detection> filt(List<Detection> cands, void Function(int) addDropped) {
    final kept = [for (final d in cands) if (!isHot(d)) d];
    addDropped(cands.length - kept.length);
    return kept;
  }

  var dropped = 0;
  void add(int n) => dropped += n;
  for (final rec in match.records) {
    rec.toss = filt(rec.toss, add);
    rec.fballs = filt(rec.fballs, add);
    rec.rballs = filt(rec.rballs, add);
  }
  match.staticSuppressed = true;
  return dropped;
}

// ===========================================================================
// Serve events (near side — slice 3b; far side lands in slice 3c)
// ===========================================================================
class ServeEvent {
  final double t;
  final String side; // "near" | "far"
  final double score;
  bool traceConfirmed;
  bool tossSeen; // a toss peaked above the server's head

  ServeEvent(this.t, this.side, this.score,
      {this.traceConfirmed = false, this.tossSeen = false});

  /// Independent evidence beyond the score: a serve-like ball trace.  Far
  /// events are trace-born, so they are always supported.
  bool get supported => traceConfirmed;
}

double _hypot(double a, double b) => math.sqrt(a * a + b * b);

/// Kalman-filtered "ball moving up, predominantly vertically, above the head"
/// detector — the near-serve toss signature.  Port of _TossTracker: seed from
/// two nearby candidates, associate by nearest-to-prediction (not raw
/// confidence), coast through short gaps, confirm on consecutive REAL
/// detections (coasting ticks don't count).  Reuses kalman.dart.
class _TossTracker {
  final SegmenterConfig cfg;
  final double confFloor;
  KalmanFilter? kf;
  double lastT = 0.0;
  double lastUpdateT = -double.infinity;
  List<double>? pending; // [t, x, y]
  int risingConsecutive = 0;
  double? risingSinceT;
  double? tossMinY;

  _TossTracker(this.cfg, this.confFloor);

  KalmanFilter _seed(double x, double y, double vx, double vy) => KalmanFilter(
        F: Mat.identity(4),
        H: Mat(2, 4)
          ..set(0, 0, 1)
          ..set(1, 1, 1),
        R: Mat.identity(2).scaled(cfg.tossKfRPx),
        Q: Mat.diag([cfg.tossKfQPos, cfg.tossKfQPos, cfg.tossKfQVel, cfg.tossKfQVel]),
        P: Mat.identity(4).scaled(100.0),
        x: Mat.colVec([x, y, vx, vy]),
      );

  void _predict(double dt) {
    if (dt < 1e-6) dt = 1e-6;
    kf!.F = Mat.identity(4)
      ..set(0, 2, dt)
      ..set(1, 3, dt);
    kf!.Q = Mat.diag(
        [cfg.tossKfQPos, cfg.tossKfQPos, cfg.tossKfQVel, cfg.tossKfQVel]).scaled(dt);
    kf!.predict();
  }

  double update(List<Detection> candidates, double headY, double now) {
    final c = [for (final d in candidates) if (d.conf >= confFloor) d];

    if (kf == null) {
      final chosen = c.isEmpty
          ? null
          : c.reduce((a, b) => b.conf > a.conf ? b : a);
      var justSeeded = false;
      if (chosen != null) {
        if (pending != null) {
          final pt = pending![0], px = pending![1], py = pending![2];
          final dt = now - pt;
          if (dt > 0 &&
              dt <= cfg.tossSeedMaxDtS &&
              _hypot(chosen.x - px, chosen.y - py) <= cfg.tossSeedGatePx) {
            kf = _seed(chosen.x, chosen.y, (chosen.x - px) / dt,
                (chosen.y - py) / dt);
            lastT = now;
            lastUpdateT = now;
            pending = null;
            justSeeded = true;
          } else {
            pending = [now, chosen.x, chosen.y];
          }
        } else {
          pending = [now, chosen.x, chosen.y];
        }
      }
      return _score(headY, now, justSeeded);
    }

    final dt = now - lastT;
    if (dt > 0) _predict(dt);
    lastT = now;

    Detection? assoc;
    if (c.isNotEmpty) {
      final px = kf!.x.d[0], py = kf!.x.d[1];
      final gated = [
        for (final d in c)
          if (_hypot(d.x - px, d.y - py) <= cfg.tossAssocGatePx) d
      ];
      if (gated.isNotEmpty) {
        assoc = gated.reduce((a, b) => b.conf > a.conf ? b : a);
      }
    }

    if (assoc != null) {
      kf!.update(Mat.colVec([assoc.x, assoc.y]));
      lastUpdateT = now;
    } else if (now - lastUpdateT > cfg.tossCoastMaxS) {
      kf = null;
      risingConsecutive = 0;
      risingSinceT = null;
    }
    return _score(headY, now, assoc != null);
  }

  double _score(double headY, double now, bool hadDetection) {
    if (kf == null) {
      risingConsecutive = 0;
      risingSinceT = null;
      return 0.0;
    }
    final y = kf!.x.d[1], vx = kf!.x.d[2], vy = kf!.x.d[3];
    final aboveHead = y < headY;
    final rising =
        vy <= -cfg.tossMinVyPxS && vx.abs() <= cfg.tossMaxHorizRatio * vy.abs();
    if (aboveHead && (tossMinY == null || y < tossMinY!)) tossMinY = y;
    if (aboveHead && rising && hadDetection) {
      if (risingConsecutive == 0) risingSinceT = now;
      risingConsecutive += 1;
    } else {
      risingConsecutive = 0;
      risingSinceT = null;
    }
    final durationOk = risingSinceT != null &&
        now - risingSinceT! >= cfg.tossMinRiseDurationS;
    if (risingConsecutive >= cfg.tossConfirmFrames && durationOk) return 1.0;
    if (risingConsecutive >= 1) return 0.5;
    return 0.0;
  }
}

/// Toss (Kalman) + trophy-pose weighted near-serve scoring (port of
/// _NearServeScorer).
class _NearServeScorer {
  final SegmenterConfig cfg;
  late _TossTracker toss;
  final List<List<double>> _trophyScores = []; // [score, t]
  final List<List<double>> _tossScores = [];

  _NearServeScorer(this.cfg) {
    reset();
  }

  void reset() {
    toss = _TossTracker(cfg, cfg.tossConfFloor);
    _trophyScores.clear();
    _tossScores.clear();
  }

  double update(FrameRecord rec, double now) {
    if (rec.nearBox == null) return 0.0;
    if (rec.trophy > 0) _trophyScores.add([rec.trophy, now]);
    final ts = toss.update(rec.toss, rec.nearBox![1].toDouble(), now);
    if (ts > 0) _tossScores.add([ts, now]);
    for (final buf in [_trophyScores, _tossScores]) {
      while (buf.isNotEmpty && now - buf.first[1] > cfg.serveEventWindowS) {
        buf.removeAt(0);
      }
    }
    final maxTrophy =
        _trophyScores.fold<double>(0.0, (m, e) => e[0] > m ? e[0] : m);
    final maxToss = _tossScores.fold<double>(0.0, (m, e) => e[0] > m ? e[0] : m);
    return cfg.trophyWeight * maxTrophy + cfg.tossWeight * maxToss;
  }

  bool validate(FrameRecord rec) {
    if (rec.nearBox == null) return false;
    return toss.tossMinY != null && toss.tossMinY! < rec.nearBox![1];
  }
}

/// Near-serve events: ready-band dwell + toss/trophy score, with the
/// band-flicker grace window.  Port of detect_serve_events (near path).
/// (Far side lands in slice 3c.)
List<ServeEvent> detectNearServeEvents(
    MatchTelemetry match, SegmenterConfig cfg) {
  final scorer = _NearServeScorer(cfg);
  final events = <ServeEvent>[];
  double? readyStart;
  var armed = false;
  final bandHist = <List<double>>[]; // [t, inBand?1:0]
  var cooldownUntil = -double.infinity;
  double? bandExitT;

  for (final rec in match.records) {
    final now = rec.t;
    if (now < cooldownUntil) continue;

    var inBand = false;
    if (rec.nearWorld != null) {
      final dist = -rec.nearWorld![1]; // behind the near baseline (y=0)
      inBand = cfg.nearBandFt[0] <= dist && dist <= cfg.nearBandFt[1];
    }

    if (!armed) {
      if (inBand) {
        readyStart ??= now;
        if (now - readyStart > cfg.readyDwellS) {
          armed = true;
          scorer.reset();
          bandHist.clear();
        }
      } else {
        readyStart = null;
      }
      continue;
    }

    // armed: band-flicker grace + out-of-band ratio, then score.
    if (inBand) {
      bandExitT = null;
    } else {
      bandExitT ??= now;
    }
    final inGrace = inBand || (now - bandExitT! <= cfg.tossBandGraceS);

    bandHist.add([now, inBand ? 1.0 : 0.0]);
    while (bandHist.isNotEmpty && now - bandHist.first[0] > cfg.bandWindowS) {
      bandHist.removeAt(0);
    }
    var ratioExceeded = false;
    if (bandHist.length > 1) {
      final total = bandHist.last[0] - bandHist.first[0];
      if (total > 1.0) {
        var tOut = 0.0;
        for (var i = 0; i < bandHist.length - 1; i++) {
          if (bandHist[i][1] == 0.0) tOut += bandHist[i + 1][0] - bandHist[i][0];
        }
        ratioExceeded = tOut / total > cfg.bandOutRatio;
      }
    }

    if (ratioExceeded && !inGrace) {
      armed = false;
      readyStart = null;
      continue;
    }
    if (!inGrace) continue;

    final score = scorer.update(rec, now);
    if (score >= cfg.serveScoreThreshold && scorer.validate(rec)) {
      events.add(ServeEvent(now, 'near', score, tossSeen: true));
      armed = false;
      readyStart = null;
      cooldownUntil = now + cfg.serveRearmS;
    }
  }
  return events;
}

// ===========================================================================
// Ball-trace replay (slice 3c) — reuses ball_tracker.dart's IMM tracker
// ===========================================================================
class _SmoothedVelocity {
  final double windowSec;
  final List<List<double>> _pts = []; // [t, x, y]
  _SmoothedVelocity(this.windowSec);

  void add(double t, double x, double y) {
    _pts.add([t, x, y]);
    final cutoff = t - windowSec;
    while (_pts.isNotEmpty && _pts.first[0] < cutoff) {
      _pts.removeAt(0);
    }
  }

  List<double>? velocity() {
    if (_pts.length < 2) return null;
    final f = _pts.first, l = _pts.last;
    if (l[0] <= f[0]) return null;
    final dt = l[0] - f[0];
    return [(l[1] - f[1]) / dt, (l[2] - f[2]) / dt];
  }
}

bool _isCarried(
    List<double>? vBall, List<double>? vPlayer, SegmenterConfig cfg) {
  if (vBall == null || vPlayer == null) return false;
  final ballSpeed = _hypot(vBall[0], vBall[1]);
  final playerSpeed = _hypot(vPlayer[0], vPlayer[1]);
  if (ballSpeed < 1e-6) return false;
  final ratio = _hypot(vBall[0] - vPlayer[0], vBall[1] - vPlayer[1]) / ballSpeed;
  return ratio < cfg.couplingRatioMax &&
      playerSpeed >= cfg.couplingMinPlayerSpeed;
}

class ReplayFrame {
  final double t;
  final bool genuine; // trace alive, above velocity floor, not carried
  final double racketProb;
  final List<double>? position;
  final double tsd; // seconds since the last real detection
  final double bounceProb;
  final List<double>? det; // raw detection associated this frame, or null
  ReplayFrame(this.t, this.genuine, this.racketProb, this.position, this.tsd,
      this.bounceProb, this.det);
}

/// Re-run the IMM ball tracker over the recorded detections in [t0, t1).
/// Port of replay_ball_tracker/_replay_core (collect=false).  In-box
/// detections are dropped (racket/arm false positives) unless the tracked ball
/// was already predicted there (inbox_accept — the contact moment).  Merges
/// the native-res channels (fballs, rballs).
List<ReplayFrame> replayBallTracker(
    MatchTelemetry match, double t0, double t1, SegmenterConfig cfg) {
  final persp = makeImageRowPerspective(cfg.frameHeightPx);
  final mgr = BallTrackManager(match.fps, perspectiveScale: persp);
  final ballVel = _SmoothedVelocity(cfg.couplingWindowS);
  final playerVel = _SmoothedVelocity(cfg.couplingWindowS);

  final out = <ReplayFrame>[];
  for (final rec in match.slice(t0, t1)) {
    final tp = mgr.track?.position; // previous-frame filtered position
    var inboxAccept = 0.0;
    if (tp != null && cfg.inboxAcceptPx > 0) {
      inboxAccept = cfg.inboxAcceptPx * math.max(persp(tp[1]), 0.35);
    }

    final dets = <Detection>[];
    for (final d in [...rec.balls, ...rec.fballs, ...rec.rballs]) {
      final bx = d.x, by = d.y;
      var inside = false;
      for (final box in [rec.nearBox, rec.farBox]) {
        if (box != null &&
            box[0] <= bx &&
            bx <= box[2] &&
            box[1] <= by &&
            by <= box[3]) {
          inside = true;
          break;
        }
      }
      if (inside &&
          !(inboxAccept > 0 &&
              tp != null &&
              _hypot(bx - tp[0], by - tp[1]) <= inboxAccept)) {
        continue; // racket/arm/body false positive
      }
      if (dets.any((e) => (bx - e.x).abs() <= 6.0 && (by - e.y).abs() <= 6.0)) {
        continue; // both passes saw the same ball
      }
      dets.add(Detection(bx, by, d.conf));
    }

    final status = mgr.update(dets, rec.t);

    if (status.position != null) {
      ballVel.add(rec.t, status.position![0], status.position![1]);
    }
    if (rec.nearBox != null) {
      playerVel.add(rec.t, (rec.nearBox![0] + rec.nearBox![2]) / 2.0,
          (rec.nearBox![1] + rec.nearBox![3]) / 2.0);
    }

    var genuine = false;
    if (status.hasMovingTrace && status.position != null) {
      final floor = cfg.moveVelocityFloorPxS * persp(status.position![1]);
      if (status.speedPxS >= floor) {
        genuine = !_isCarried(ballVel.velocity(), playerVel.velocity(), cfg);
      }
    }

    List<double>? assoc;
    if (dets.isNotEmpty &&
        status.position != null &&
        status.timeSinceDetection < 0.75 / math.max(match.fps, 1e-6)) {
      Detection? best;
      var bestD = double.infinity;
      for (final d in dets) {
        final ddx = d.x - status.position![0], ddy = d.y - status.position![1];
        final dd = ddx * ddx + ddy * ddy;
        if (dd < bestD) {
          bestD = dd;
          best = d;
        }
      }
      if (best != null) assoc = [best.x, best.y];
    }
    out.add(ReplayFrame(rec.t, genuine, status.racketProb, status.position,
        status.timeSinceDetection, status.bounceProb, assoc));
  }
  return out;
}

/// Maximal [start, end] intervals of genuine trace motion, folding gaps
/// <= mergeGapS.  Ends anchored to the last real detection (t - tsd).
List<List<double>> aliveIntervals(List<ReplayFrame> replay, double mergeGapS) {
  final raw = <List<double>>[]; // [start, rawEnd, anchoredEnd]
  for (final fr in replay) {
    if (!fr.genuine) continue;
    final anchored = fr.t - math.min(fr.tsd, fr.t);
    if (raw.isNotEmpty && fr.t - raw.last[1] <= mergeGapS) {
      raw.last[1] = fr.t;
      raw.last[2] = math.max(raw.last[2], anchored);
    } else {
      raw.add([fr.t, fr.t, math.max(fr.t - fr.tsd, 0.0)]);
    }
  }
  return [for (final r in raw) [r[0], math.max(r[0], r[2])]];
}

/// A serve-like trace: net downward + horizontal motion over any ~0.3s
/// stretch of genuine points (perspective-scaled).
bool confirmServeTrace(List<ReplayFrame> replay, SegmenterConfig cfg) {
  final persp = makeImageRowPerspective(cfg.frameHeightPx);
  final pts = [
    for (final fr in replay)
      if (fr.genuine && fr.position != null)
        [fr.t, fr.position![0], fr.position![1]]
  ];
  for (var i = 1; i < pts.length; i++) {
    final t1 = pts[i][0], x1 = pts[i][1], y1 = pts[i][2];
    var j = i - 1;
    while (j > 0 && t1 - pts[j][0] < 0.3) {
      j--;
    }
    final t0 = pts[j][0], x0 = pts[j][1], y0 = pts[j][2];
    final dt = t1 - t0;
    if (dt < 0.15) continue;
    final scale = persp((y0 + y1) / 2.0);
    if ((y1 - y0) / dt >= cfg.traceDownwardPxS * scale &&
        (x1 - x0).abs() / dt >= cfg.traceHorizontalPxS * scale) {
      return true;
    }
  }
  return false;
}

// ===========================================================================
// Far-serve detection (slice 3c)
// ===========================================================================

/// Largest image-y a far-serve trace stretch may START at — calibrated from
/// observed far-player feet (median + pad), else a frame fraction.
double farRegionCutoffY(MatchTelemetry match, SegmenterConfig cfg) {
  final feet = [
    for (final r in match.records)
      if (r.farBox != null) r.farBox![3].toDouble()
  ]..sort();
  if (feet.length >= cfg.farFeetMinSamples) {
    return feet[feet.length ~/ 2] + cfg.farOriginPadPx;
  }
  return cfg.frameHeightPx * cfg.farTraceOriginFrac;
}

bool _farServeStretch(List<List<double>> pts, SegmenterConfig cfg,
    double Function(double) persp, double yOriginMax) {
  for (var i = 1; i < pts.length; i++) {
    final t1 = pts[i][0], x1 = pts[i][1], y1 = pts[i][2];
    var j = i - 1;
    while (j > 0 && t1 - pts[j][0] < 0.3) {
      j--;
    }
    final t0 = pts[j][0], x0 = pts[j][1], y0 = pts[j][2];
    final dt = t1 - t0;
    if (dt < 0.15 || y0 > yOriginMax) continue;
    final scale = persp((y0 + y1) / 2.0);
    if ((y1 - y0) / dt >= cfg.traceDownwardPxS * scale &&
        (x1 - x0).abs() / dt >= cfg.traceHorizontalPxS * scale) {
      return true;
    }
  }
  return false;
}

/// Far-serve trace onsets: ORIGIN (far region) + MOTION (down+horizontal,
/// early) + QUIET (dead time before, micro-blips ignored).
List<double> farServeTraceOnsets(MatchTelemetry match, SegmenterConfig cfg) {
  final replay = replayBallTracker(match, 0.0, match.duration + 1.0, cfg);
  final intervals = aliveIntervals(replay, cfg.aliveMergeGapS);
  final persp = makeImageRowPerspective(cfg.frameHeightPx);
  final cutoffY = farRegionCutoffY(match, cfg);

  final onsets = <double>[];
  var prevEnd = -double.infinity;
  for (final iv in intervals) {
    final start = iv[0], end = iv[1];
    final quietOk = start - prevEnd >= cfg.farTraceQuietS;
    if (end - start >= cfg.farTraceMinIntervalS) {
      prevEnd = math.max(prevEnd, end);
    }
    if (!quietOk) continue;
    final headEnd = math.min(end, start + cfg.farTraceHeadS);
    final pts = [
      for (final fr in replay)
        if (fr.genuine &&
            fr.position != null &&
            start <= fr.t &&
            fr.t <= headEnd)
          [fr.t, fr.position![0], fr.position![1]]
    ];
    if (_farServeStretch(pts, cfg, persp, cutoffY)) onsets.add(start);
  }
  return onsets;
}

/// A logged far near-miss (for review); does not affect segmentation.
class FarMiss {
  final double t;
  final double score;
  final String reason;
  const FarMiss(this.t, this.score, this.reason);
}

/// Far serves from the trace alone, gated on far-player PRESENCE (a tracked
/// far box within farPresenceSlackS of the onset).
List<ServeEvent> _detectFarServeEvents(
    MatchTelemetry match, SegmenterConfig cfg, List<FarMiss>? farMisses) {
  final farTs = [
    for (final r in match.records)
      if (r.farBox != null) r.t
  ];
  final events = <ServeEvent>[];
  for (final onset in farServeTraceOnsets(match, cfg)) {
    final i = _bisectLeft(farTs, onset - cfg.farPresenceSlackS);
    final present = i < farTs.length && farTs[i] <= onset + cfg.farPresenceSlackS;
    if (present) {
      events.add(ServeEvent(onset, 'far', 1.0, traceConfirmed: true));
    } else {
      farMisses?.add(FarMiss(onset, 1.0, 'no_far_player'));
    }
  }
  return events;
}

/// Serve events for one side (dispatcher — mirrors detect_serve_events).
List<ServeEvent> detectServeEvents(
    MatchTelemetry match, String side, SegmenterConfig cfg,
    {List<FarMiss>? farMisses}) {
  if (side == 'far') return _detectFarServeEvents(match, cfg, farMisses);
  return detectNearServeEvents(match, cfg);
}

// ===========================================================================
// Dedupe + serving-side HMM (slice 3c)
// ===========================================================================
List<ServeEvent> dedupeServeEvents(
    List<ServeEvent> events, SegmenterConfig cfg) {
  final sorted = [...events]..sort((a, b) => a.t.compareTo(b.t));
  final kept = <ServeEvent>[];
  for (final evt in sorted) {
    if (kept.isNotEmpty &&
        evt.t - kept.last.t < cfg.minServeSeparationS) {
      final prev = kept.last;
      if (evt.supported != prev.supported) {
        if (evt.supported) kept[kept.length - 1] = evt; // supported wins
        continue;
      }
      if (prev.side != evt.side) {
        if (evt.side == 'near') kept[kept.length - 1] = evt; // near wins
        continue;
      }
      continue; // earlier wins
    }
    kept.add(evt);
  }
  return kept;
}

List<String> _viterbi(List<String> obsSides, double pStay, double pCorrect) {
  final n = obsSides.length;
  if (n == 0) return [];
  if (n == 1) return [...obsSides];
  final sides = ['near', 'far'];
  final logTrans = [
    [math.log(pStay), math.log(1 - pStay)],
    [math.log(1 - pStay), math.log(pStay)],
  ];
  final logEmit = [
    [math.log(pCorrect), math.log(1 - pCorrect)],
    [math.log(1 - pCorrect), math.log(pCorrect)],
  ];
  final obs = [for (final o in obsSides) o == 'near' ? 0 : 1];
  var delta = [
    math.log(0.5) + logEmit[0][obs[0]],
    math.log(0.5) + logEmit[1][obs[0]],
  ];
  final psi = [for (var i = 0; i < n; i++) [0, 0]];
  for (var t = 1; t < n; t++) {
    final newDelta = [0.0, 0.0];
    for (var s = 0; s < 2; s++) {
      // scores[from] = delta[from] + logTrans[from][s]
      var best = 0;
      var bestScore = delta[0] + logTrans[0][s];
      final s1 = delta[1] + logTrans[1][s];
      if (s1 > bestScore) {
        best = 1;
        bestScore = s1;
      }
      psi[t][s] = best;
      newDelta[s] = bestScore + logEmit[s][obs[t]];
    }
    delta = newDelta;
  }
  final path = List<int>.filled(n, 0);
  path[n - 1] = delta[1] > delta[0] ? 1 : 0;
  for (var t = n - 2; t >= 0; t--) {
    path[t] = psi[t + 1][path[t + 1]];
  }
  return [for (final s in path) sides[s]];
}

/// Drop side-anomalous events that lack a confirming trace (weak noise).
/// Any trace-confirmed event is shielded regardless of side.
List<ServeEvent> hmmFilterEvents(List<ServeEvent> events, SegmenterConfig cfg) {
  if (events.length < 2) return events;
  final decoded =
      _viterbi([for (final e in events) e.side], cfg.hmmPStay, cfg.hmmPCorrect);
  final kept = <ServeEvent>[];
  for (var i = 0; i < events.length; i++) {
    final evt = events[i];
    if (evt.side != decoded[i] && !evt.supported) continue;
    kept.add(evt);
  }
  return kept;
}

// ===========================================================================
// Player kinematics + point-end fusion + orchestration (slice 3d)
// ===========================================================================
int _bisectRight(List<double> a, double x) {
  var lo = 0, hi = a.length;
  while (lo < hi) {
    final mid = (lo + hi) >> 1;
    if (x < a[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

/// Near-player smoothed speed + direction-reversal ("rally cue") timeline.
/// Port of PlayerKinematics (near player only; far tracking too unreliable).
class PlayerKinematics {
  final List<double?> speedNear;
  final List<double> rallyCues;
  PlayerKinematics._(this.speedNear, this.rallyCues);

  factory PlayerKinematics(MatchTelemetry match, SegmenterConfig cfg) {
    final n = match.ts.length;
    final speedNear = List<double?>.filled(n, null);
    final reversalTimes = <double>[];

    final valid = <List<double>>[]; // [t, wx, wy, idx]
    for (var i = 0; i < match.records.length; i++) {
      final r = match.records[i];
      if (r.nearWorld != null) {
        valid.add([r.t, r.nearWorld![0], r.nearWorld![1], i.toDouble()]);
      }
    }

    var j = 0;
    var lastSign = 0;
    for (var k = 0; k < valid.length; k++) {
      final t1 = valid[k][0], x1 = valid[k][1], y1 = valid[k][2];
      final idx = valid[k][3].toInt();
      while (j < k && t1 - valid[j][0] > cfg.speedWindowS) {
        j++;
      }
      final jj =
          (j > 0 && t1 - valid[j][0] < cfg.speedMinDtS) ? math.max(0, j - 1) : j;
      final t0 = valid[jj][0], x0 = valid[jj][1], y0 = valid[jj][2];
      final dt = t1 - t0;
      if (dt < cfg.speedMinDtS) continue;
      final vx = (x1 - x0) / dt, vy = (y1 - y0) / dt;
      speedNear[idx] = _hypot(vx, vy);
      if (vx.abs() >= cfg.reversalSpeedFtS) {
        final sign = vx > 0 ? 1 : -1;
        if (lastSign != 0 && sign != lastSign) reversalTimes.add(t1);
        lastSign = sign;
      }
    }
    return PlayerKinematics._(speedNear, reversalTimes);
  }

  bool rallyLike(double t0, double t1) {
    final i = _bisectRight(rallyCues, t0);
    return i < rallyCues.length && rallyCues[i] < t1;
  }

  double chainActivity(double seedT, double capT, double gapS) {
    var last = seedT;
    var i = _bisectRight(rallyCues, seedT);
    while (i < rallyCues.length && rallyCues[i] <= capT) {
      if (rallyCues[i] - last <= gapS) {
        last = rallyCues[i];
        i++;
      } else {
        break;
      }
    }
    return last;
  }
}

class PointEnd {
  final double endT;
  final String method;
  const PointEnd(this.endT, this.method);
}

/// Estimate when the point starting at [serveT] ended, searching only inside
/// (serveT, tNext).  Port of find_point_end: trace-chain anchored at the serve,
/// extended/replaced by rally-cue player activity.
PointEnd findPointEnd(MatchTelemetry match, PlayerKinematics kin, double serveT,
    double tNext, SegmenterConfig cfg,
    {List<ReplayFrame>? replay}) {
  var cap = math.min(math.min(tNext - cfg.nextServeGuardS, serveT + cfg.maxPointS),
      match.duration);
  cap = math.max(cap, serveT + cfg.minPointS);

  final rep = replay ?? replayBallTracker(match, serveT - 0.3, cap, cfg);
  final intervals = aliveIntervals(rep, cfg.aliveMergeGapS);

  // 1. Trace chain anchored at the serve.
  double? chainEnd;
  List<double>? startIv;
  for (final iv in intervals) {
    if (serveT - 0.5 <= iv[0] && iv[0] <= serveT + cfg.serveChainWindowS) {
      startIv = iv;
      break;
    }
  }
  if (startIv != null) {
    var ce = math.max(startIv[1], serveT);
    for (final iv in intervals) {
      if (iv[0] <= ce) {
        ce = math.max(ce, iv[1]);
        continue;
      }
      final gap = iv[0] - ce;
      if (gap <= cfg.chainGapS) {
        ce = iv[1];
      } else if (gap <= cfg.chainGapActiveS && kin.rallyLike(ce, iv[0])) {
        ce = iv[1];
      } else {
        break;
      }
    }
    chainEnd = ce;
  }

  // 2. Player-activity chaining.
  double end;
  String method;
  if (chainEnd != null) {
    var actEnd = kin.chainActivity(chainEnd, cap, cfg.activityGapS);
    actEnd = math.min(actEnd, chainEnd + cfg.activityExtendMaxS);
    end = math.max(chainEnd, actEnd);
    method = actEnd <= chainEnd ? 'trace' : 'trace+activity';
  } else {
    final actEnd = kin.chainActivity(serveT, cap, cfg.activityGapS);
    if (actEnd > serveT) {
      end = actEnd;
      method = 'activity';
    } else {
      end = serveT + cfg.fallbackPointS;
      method = 'fallback';
    }
  }

  end = math.min(math.max(end, serveT + cfg.minPointS), cap);
  return PointEnd(end, method);
}

class PointSegment {
  final int point;
  final String side;
  final double serveT;
  final double endT;
  final double start; // serveT - pre_roll (clamped)
  final double end; // endT + end_pad (clamped)
  final String endMethod;
  final bool traceConfirmed;
  final double score;
  const PointSegment({
    required this.point,
    required this.side,
    required this.serveT,
    required this.endT,
    required this.start,
    required this.end,
    required this.endMethod,
    required this.traceConfirmed,
    required this.score,
  });
}

/// Full stage-2 segmentation: telemetry → point segments.  Port of
/// segment_match — the top-level entry the cutter/UI calls.
List<PointSegment> segmentMatch(MatchTelemetry match,
    {SegmenterConfig? config, List<FarMiss>? farMissesOut}) {
  final cfg = config ?? SegmenterConfig();
  if (match.meta.isNotEmpty) {
    final cl = match.meta['court_length_ft'];
    if (cl != null) cfg.courtLengthFt = (cl as num).toDouble();
    final size = match.meta['analysis_size'] as List?;
    if (size != null) cfg.frameHeightPx = (size[1] as num).toDouble();
  }

  suppressStaticCandidates(match, cfg);
  final farMisses = farMissesOut ?? <FarMiss>[];
  final near = detectServeEvents(match, 'near', cfg);
  final far = detectServeEvents(match, 'far', cfg, farMisses: farMisses);

  final kin = PlayerKinematics(match, cfg);

  // Near-serve trace confirmation BEFORE dedupe (a confirmed real serve can
  // displace the unconfirmed aborted-toss candidate).  Far events are
  // trace-born (confirmed by construction).
  final candidates = [...near, ...far]..sort((a, b) => a.t.compareTo(b.t));
  for (final evt in candidates) {
    if (evt.side == 'near') {
      final rep = replayBallTracker(
          match, evt.t - 0.3, evt.t + cfg.confirmWindowS, cfg);
      evt.traceConfirmed = confirmServeTrace(rep, cfg);
    }
  }

  var events = dedupeServeEvents(candidates, cfg);
  events = hmmFilterEvents(events, cfg);
  if (events.isEmpty) return [];

  final segments = <PointSegment>[];
  for (var i = 0; i < events.length; i++) {
    final evt = events[i];
    final tNext = i + 1 < events.length ? events[i + 1].t : match.duration;
    final pe = findPointEnd(match, kin, evt.t, tNext, cfg);
    final pre = evt.side == 'far' ? cfg.farPreRollS : cfg.preRollS;
    segments.add(PointSegment(
      point: i + 1,
      side: evt.side,
      serveT: evt.t,
      endT: pe.endT,
      start: math.max(0.0, evt.t - pre),
      end: math.min(match.duration, pe.endT + cfg.endPadS),
      endMethod: pe.method,
      traceConfirmed: evt.traceConfirmed,
      score: evt.score,
    ));
  }
  return segments;
}
