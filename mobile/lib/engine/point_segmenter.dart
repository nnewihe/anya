import 'dart:convert';
import 'dart:math' as math;

import 'ball_tracker.dart' show Detection;
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
