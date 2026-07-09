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
}
