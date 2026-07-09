import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/ball_tracker.dart' show Detection;
import 'package:rally_predictor/engine/point_segmenter.dart';

/// Stage-2 foundation (slice 3a): JSONL loading + static-candidate suppression.

FrameRecord _rec(int f, double t,
        {List<Detection> toss = const [], List<Detection> fballs = const []}) =>
    FrameRecord(
      f: f,
      t: t,
      nearBox: null,
      nearWorld: null,
      farBox: null,
      farHeld: false,
      farWorld: null,
      balls: const [],
      toss: toss,
      fballs: fballs,
      rballs: const [],
      trophy: 0.0,
      stgcn: 0.0,
    );

/// Mirrors _mk_rec in point_segmenter.py's self-test: near box (430,350,490,500)
/// when nearWy is given, far box (450,120,480,175) when farWy is given.
FrameRecord _mkRec(int f, double t,
    {double? nearWy,
    double? farWy,
    double nearWx = 13.5,
    double farWx = 13.5,
    List<Detection> toss = const [],
    double trophy = 0.0}) {
  return FrameRecord(
    f: f,
    t: t,
    nearBox: nearWy != null ? [430, 350, 490, 500] : null,
    nearWorld: nearWy != null ? [nearWx, nearWy] : null,
    farBox: farWy != null ? [450, 120, 480, 175] : null,
    farHeld: false,
    farWorld: farWy != null ? [farWx, farWy] : null,
    balls: const [],
    toss: toss,
    fballs: const [],
    rballs: const [],
    trophy: trophy,
    stgcn: 0.0,
  );
}

MatchTelemetry _match(List<FrameRecord> recs) => MatchTelemetry(const {
      'fps': 30.0,
      'stride': 1,
      'court_length_ft': 78.0,
      'analysis_size': [960, 540],
    }, recs);

const double _dt = 1.0 / 30.0;

void main() {
  test('loadTelemetry parses meta + records (stage-1 schema)', () {
    const jsonl = '{"meta":{"version":4,"fps":59.94,"stride":1,'
        '"analysis_size":[960,540],"court_length_ft":78.0}}\n'
        '{"f":0,"t":0.0,"np":[430,350,490,500],"npw":[13.5,-1.5],'
        '"fp":null,"fph":0,"fpw":null,"balls":[[100.0,200.0,0.8]],'
        '"toss":[],"fballs":[],"trophy":0.0,"stgcn":0.0}\n'
        '{"f":2,"t":0.0334,"np":null,"npw":null,"fp":[450,120,480,175],'
        '"fph":1,"fpw":[13.9,80.0],"balls":[],"toss":[[455.0,300.0,0.9]],'
        '"fballs":[[500.0,130.0,0.7]],"trophy":0.1,"stgcn":0.2}\n';
    final m = loadTelemetry(jsonl);

    expect(m.records.length, 2);
    expect(m.fps, closeTo(59.94, 1e-6)); // fps / stride
    expect(m.duration, closeTo(0.0334, 1e-9));

    final r0 = m.records[0];
    expect(r0.nearBox, [430, 350, 490, 500]);
    expect(r0.nearWorld, [13.5, -1.5]);
    expect(r0.farBox, isNull);
    expect(r0.farHeld, false);
    expect(r0.balls.length, 1);
    expect(r0.balls.first.x, 100.0);
    expect(r0.balls.first.conf, 0.8);

    final r1 = m.records[1];
    expect(r1.nearBox, isNull);
    expect(r1.farBox, [450, 120, 480, 175]);
    expect(r1.farHeld, true);
    expect(r1.farWorld, [13.9, 80.0]);
    expect(r1.toss.length, 1);
    expect(r1.fballs.length, 1);
    expect(r1.rballs, isEmpty); // absent key -> empty
    expect(r1.trophy, 0.1);
  });

  test('indexRange / slice cover [t0, t1)', () {
    final recs = [for (var i = 0; i < 10; i++) _rec(i, i.toDouble())];
    final m = MatchTelemetry(const {'fps': 30.0, 'stride': 1}, recs);
    expect(m.indexRange(2.0, 5.0), [2, 5]); // bisectLeft
    expect(m.slice(2.0, 5.0).map((r) => r.f).toList(), [2, 3, 4]);
  });

  test('static-candidate suppression drops a stuck candidate, keeps movers',
      () {
    // 50 frames: a static toss stuck in one cell every frame (100% > 4%),
    // plus a one-off toss in a different cell on a single frame (2% < 4%).
    final recs = <FrameRecord>[];
    for (var i = 0; i < 50; i++) {
      final toss = [const Detection(700.0, 300.0, 0.95)]; // stuck
      if (i == 25) toss.add(const Detection(200.0, 200.0, 0.9)); // mover
      recs.add(_rec(i, i / 30.0, toss: toss));
    }
    final m = MatchTelemetry(const {'fps': 30.0, 'stride': 1}, recs);
    final cfg = SegmenterConfig();
    final dropped = suppressStaticCandidates(m, cfg);

    expect(dropped, 50); // the stuck candidate removed from all 50 frames
    // stuck cell gone everywhere; the single mover survives.
    var stuck = 0, mover = 0;
    for (final r in m.records) {
      for (final d in r.toss) {
        if (d.x == 700.0) stuck++;
        if (d.x == 200.0) mover++;
      }
    }
    expect(stuck, 0);
    expect(mover, 1);

    // idempotent: a second call drops nothing.
    expect(suppressStaticCandidates(m, cfg), 0);
  });

  // --- near-serve detection (slice 3b), ported from point_segmenter.py
  //     --self-test near-side checks ---

  test('near serve detected from a rising toss above the head', () {
    final recs = <FrameRecord>[];
    var f = 0;
    var t = 0.0;
    for (var i = 0; i < 45; i++) {
      recs.add(_mkRec(f, t, nearWy: -1.5)); // settled in the ready band
      f++;
      t += _dt;
    }
    var tossY = 345.0;
    for (var i = 0; i < 6; i++) {
      tossY -= 9.0; // climbing above the box top (350)
      recs.add(_mkRec(f, t, nearWy: -1.5, toss: [Detection(455.0, tossY, 0.9)]));
      f++;
      t += _dt;
    }
    for (var i = 0; i < 60; i++) {
      recs.add(_mkRec(f, t, nearWy: -1.5));
      f++;
      t += _dt;
    }
    final events = detectNearServeEvents(_match(recs), SegmenterConfig());
    expect(events.length, 1);
    expect(events.first.t, inInclusiveRange(1.4, 2.0)); // ~toss moment
    expect(events.first.side, 'near');
  });

  test('a below-head toss never fires', () {
    final recs = <FrameRecord>[];
    var f = 0;
    var t = 0.0;
    for (var i = 0; i < 45; i++) {
      recs.add(_mkRec(f, t, nearWy: -1.5));
      f++;
      t += _dt;
    }
    var lowY = 480.0;
    for (var i = 0; i < 6; i++) {
      lowY -= 5.0; // rising but still below/inside the box (>350)
      recs.add(_mkRec(f, t, nearWy: -1.5, toss: [Detection(455.0, lowY, 0.9)]));
      f++;
      t += _dt;
    }
    expect(detectNearServeEvents(_match(recs), SegmenterConfig()), isEmpty);
  });

  test('toss confirms despite a concurrent ready-band flicker', () {
    // Rising toss while the near-player WORLD position reads out of band —
    // the grace window must keep scoring alive (folder-68 t=163.9 bug).
    final recs = <FrameRecord>[];
    var f = 0;
    var t = 0.0;
    for (var i = 0; i < 45; i++) {
      recs.add(_mkRec(f, t, nearWy: -1.5)); // dwell in-band, arm
      f++;
      t += _dt;
    }
    var tossY = 345.0;
    for (var i = 0; i < 8; i++) {
      tossY -= 9.0;
      recs.add(_mkRec(f, t,
          nearWy: 10.0, toss: [Detection(455.0, tossY, 0.9)])); // OUT of band
      f++;
      t += _dt;
    }
    for (var i = 0; i < 30; i++) {
      recs.add(_mkRec(f, t, nearWy: -1.5));
      f++;
      t += _dt;
    }
    expect(detectNearServeEvents(_match(recs), SegmenterConfig()).length, 1);
  });

  test('detector re-arms after an aborted toss', () {
    final recs = <FrameRecord>[];
    var f = 0;
    var t = 0.0;
    void dwell(int n) {
      for (var i = 0; i < n; i++) {
        recs.add(_mkRec(f, t, nearWy: -1.5));
        f++;
        t += _dt;
      }
    }

    void tossBurst() {
      var ty = 345.0;
      for (var i = 0; i < 6; i++) {
        ty -= 9.0;
        recs.add(_mkRec(f, t, nearWy: -1.5, toss: [Detection(455.0, ty, 0.9)]));
        f++;
        t += _dt;
      }
    }

    dwell(45);
    tossBurst(); // ~1.5s (aborted-toss candidate)
    while (t < 6.5) {
      recs.add(_mkRec(f, t, nearWy: -1.5));
      f++;
      t += _dt;
    }
    tossBurst(); // ~6.5s (the real serve)
    dwell(30);
    // Both must be captured (the old 8s cooldown swallowed the second).
    expect(detectNearServeEvents(_match(recs), SegmenterConfig()).length, 2);
  });

  // --- dedupe + serving-side HMM (slice 3c), ported from the Python
  //     --self-test checks ---

  test('dedupe keeps 2 of 3 conflicting events, prefers the near one', () {
    final ded = dedupeServeEvents([
      ServeEvent(10.0, 'far', 0.9),
      ServeEvent(13.0, 'near', 0.8),
      ServeEvent(40.0, 'far', 0.9),
    ], SegmenterConfig());
    expect(ded.length, 2);
    expect(ded[0].side, 'near');
  });

  test('dedupe prefers the trace-confirmed serve of an aborted-toss pair', () {
    final ded = dedupeServeEvents([
      ServeEvent(10.0, 'near', 0.8, traceConfirmed: false),
      ServeEvent(15.0, 'near', 0.8, traceConfirmed: true),
    ], SegmenterConfig());
    expect(ded.length, 1);
    expect(ded[0].t, 15.0);
  });

  test('dedupe keeps the earlier of two confirmed events', () {
    final ded = dedupeServeEvents([
      ServeEvent(10.0, 'near', 0.8, traceConfirmed: true),
      ServeEvent(13.0, 'near', 0.8, traceConfirmed: true),
    ], SegmenterConfig());
    expect(ded.length, 1);
    expect(ded[0].t, 10.0);
  });

  test('HMM keeps a confirmed side-anomalous event (any side)', () {
    const sides = ['near', 'near', 'far', 'near', 'near'];
    final evs = [
      for (var i = 0; i < 5; i++)
        ServeEvent(10.0 + 20 * i, sides[i], 0.8, traceConfirmed: true)
    ];
    expect(hmmFilterEvents(evs, SegmenterConfig()).length, 5);
  });

  test('HMM drops an unconfirmed side-anomalous event', () {
    const sides = ['near', 'near', 'far', 'near', 'near'];
    final evs = [
      for (var i = 0; i < 5; i++)
        ServeEvent(10.0 + 20 * i, sides[i], 0.8,
            traceConfirmed: sides[i] == 'near')
    ];
    expect(hmmFilterEvents(evs, SegmenterConfig()).length, 4);
  });
}
