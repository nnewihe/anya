import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/ball_tracker.dart' show Detection;
import 'package:rally_predictor/engine/inference.dart' show LetterboxTransform;
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

  test('far-court crop rect matches the cv2 reference (folder 68 corners)', () {
    // Corners BL, BR, TR, TL in the calibration order; expected rect computed
    // with cv2.findHomography + perspectiveTransform (the Python reference).
    final rect = farCourtCropRect([
      [79.0, 377.0], // BL
      [816.0, 374.0], // BR
      [551.0, 275.0], // TR
      [412.0, 274.0], // TL
    ]);
    expect(rect.x1, closeTo(360.5, 1.0));
    expect(rect.y1, closeTo(222.4, 1.0)); // top extended 3.0x above far baseline
    expect(rect.x2, closeTo(594.3, 1.0));
    expect(rect.y2, closeTo(291.2, 1.0)); // net line
    // sanity: reaches well above the far baseline (274) into the contact region
    expect(rect.y1, lessThan(230.0));
    expect(rect.y2, greaterThan(274.0));
    expect(rect.x1, greaterThanOrEqualTo(0.0));
    expect(rect.x2, lessThanOrEqualTo(960.0));
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

  test('far-crop detection maps back to analysis coords (round-trip)', () {
    // Native crop = the folder-68 far rect at ~2.8x (native 2704 vs 960).
    final rect = farCourtCropRect([
      [79.0, 377.0],
      [816.0, 374.0],
      [551.0, 275.0],
      [412.0, 274.0],
    ]);
    const nativeScale = 2704.0 / 960.0;
    final cropW = (rect.width * nativeScale).round();
    final cropH = (rect.height * nativeScale).round();

    // The letterbox transform the source would compute for this crop.
    final s = math.min(960.0 / cropW, 960.0 / cropH);
    final tf = LetterboxTransform(
        s, (960 - cropW * s) / 2.0, (960 - cropH * s) / 2.0);

    // Forward: a known analysis point inside the rect -> where it lands in the
    // 960 letterbox; then mapFarCropToAnalysis must recover it.
    for (final frac in const [
      [0.25, 0.25],
      [0.5, 0.5],
      [0.8, 0.6],
    ]) {
      final ax = rect.x1 + frac[0] * rect.width;
      final ay = rect.y1 + frac[1] * rect.height;
      // analysis -> native crop px -> letterbox px
      final cnx = frac[0] * cropW, cny = frac[1] * cropH;
      final rawX = cnx * tf.scale + tf.padX, rawY = cny * tf.scale + tf.padY;
      final back = mapFarCropToAnalysis(rawX, rawY, tf, cropW, cropH, rect);
      expect(back[0], closeTo(ax, 0.5));
      expect(back[1], closeTo(ay, 0.5));
    }
    // The letterbox must keep the crop inside the 960 square.
    expect(cropW * s, lessThanOrEqualTo(960.0 + 1e-6));
    expect(cropH * s, lessThanOrEqualTo(960.0 + 1e-6));
  });
}
