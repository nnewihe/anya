import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/point_segmenter.dart';

/// Dev golden-master: the Dart stage-2 port must reproduce the Python
/// segment_match outputs frame-for-frame on real telemetry.  Reads telemetry
/// from the Anya drive and SKIPS gracefully when it is not mounted (so CI /
/// other machines pass).  Reference values are the Python detect_serve_events
/// (near) output as of this commit — regenerate if the pipeline changes.
///
/// Near-serve events are covered now (slice 3b, matched to the millisecond on
/// folders 68/23/21); far events + full segments extend this as 3c/3d land.
void main() {
  const cases = {
    '/Volumes/Anya/Data/68/match_match_telemetry.jsonl': [
      8.308, 44.979, 49.667, 84.236, 103.288, 109.344, 275.897, 281.235,
      950.731, 962.327, 992.924, 1037.052, 1058.391, 1086.469, 1089.973,
      1117.868, 1150.384, 1199.868, 1206.558, 1238.256, 1242.728, 1273.392,
      1287.023,
    ],
    '/Volumes/Anya/Data/23/snippet_match_telemetry.jsonl': <double>[],
    '/Volumes/Anya/Data/21/snippet_match_telemetry.jsonl': [
      59.259, 220.454, 250.617, 258.925, 289.523, 305.439, 308.775, 322.422,
      339.773, 356.490, 375.675,
    ],
  };

  for (final entry in cases.entries) {
    final path = entry.key, expected = entry.value;
    final name = path.split('/')[4];
    test('golden-master near events: folder $name', () {
      final f = File(path);
      if (!f.existsSync()) {
        markTestSkipped('telemetry not on disk');
        return;
      }
      final m = loadTelemetry(f.readAsStringSync());
      final cfg = SegmenterConfig();
      suppressStaticCandidates(m, cfg);
      final ev = detectNearServeEvents(m, cfg);
      final got = [for (final e in ev) e.t];
      // ignore: avoid_print
      print('folder $name: dart ${got.length} vs python ${expected.length}');
      // ignore: avoid_print
      print('  dart:   ${got.map((t) => t.toStringAsFixed(3)).join(",")}');
      expect(got.length, expected.length,
          reason: 'event count mismatch on $name');
      for (var i = 0; i < expected.length; i++) {
        expect(got[i], closeTo(expected[i], 0.05),
            reason: 'event $i time mismatch on $name');
      }
    });
  }
}
