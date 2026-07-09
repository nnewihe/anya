import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/ball_tracker.dart' show Detection;
import 'package:rally_predictor/engine/match_telemetry.dart';

/// The stage-1 telemetry record must serialize to the SAME JSONL schema as
/// pipeline/match_telemetry.py, or the golden-master cross-check (run Python
/// stage 1 and this side by side, diff the JSONL) is meaningless.  Locks key
/// order, rounding, and null handling.  Values are chosen off exact rounding
/// half-points so the assertions are unambiguous.
void main() {
  test('populated record matches the Python JSONL schema', () {
    const rec = MatchFrameRecord(
      f: 300,
      t: 1.23456,
      nearBox: [430, 350, 490, 500],
      nearWorld: [13.454, -1.236],
      farBox: [450, 120, 480, 175],
      farHeld: true,
      farWorld: [13.902, 80.118],
      balls: [Detection(100.04, 200.06, 0.8763)],
      toss: [Detection(455.02, 118.94, 0.902)],
      fballs: [],
    );
    final json = jsonDecode(MatchTelemetryExtractor.encodeRecord(rec));

    // Key set + order (Dart Map + jsonEncode preserve insertion order, as does
    // Python json.dumps).
    expect(json.keys.toList(), [
      'f', 't', 'np', 'npw', 'fp', 'fph', 'fpw',
      'balls', 'toss', 'fballs', 'trophy', 'stgcn',
    ]);

    expect(json['f'], 300);
    expect(json['t'], 1.2346); // round(t, 4)
    expect(json['np'], [430, 350, 490, 500]); // ints
    expect(json['npw'], [13.45, -1.24]); // round(_, 2)
    expect(json['fp'], [450, 120, 480, 175]);
    expect(json['fph'], 1); // far_held -> 1
    expect(json['fpw'], [13.9, 80.12]);
    expect(json['balls'], [
      [100.0, 200.1, 0.876] // round(cx,1) round(cy,1) round(conf,3)
    ]);
    expect(json['toss'], [
      [455.0, 118.9, 0.902]
    ]);
    expect(json['fballs'], []);
    expect(json['trophy'], 0.0); // model not ported
    expect(json['stgcn'], 0.0);
  });

  test('nulls serialize as JSON null (no player tracked)', () {
    const rec = MatchFrameRecord(
      f: 0,
      t: 0.0,
      nearBox: null,
      nearWorld: null,
      farBox: null,
      farHeld: false,
      farWorld: null,
      balls: [],
      toss: [],
      fballs: [],
    );
    final json = jsonDecode(MatchTelemetryExtractor.encodeRecord(rec));
    expect(json['np'], isNull);
    expect(json['npw'], isNull);
    expect(json['fp'], isNull);
    expect(json['fph'], 0);
    expect(json['fpw'], isNull);
    expect(json['balls'], []);
  });

  test('meta header matches the Python meta line', () {
    final meta = MatchTelemetryExtractor.metaHeader(59.94, 25193,
        hasFarBalls: false)['meta'] as Map<String, dynamic>;
    expect(meta['version'], 4);
    expect(meta['fps'], 59.94);
    expect(meta['total_frames'], 25193);
    expect(meta['stride'], 1);
    expect(meta['analysis_size'], [960, 540]);
    expect(meta['court_length_ft'], 78.0);
    expect(meta['has_trophy'], false);
    expect(meta['has_far_serve'], false);
    expect(meta['has_far_balls'], false);
  });
}
