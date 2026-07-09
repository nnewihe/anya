import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/point_segmenter.dart';

/// Dev golden-master: the Dart stage-2 port must reproduce the Python
/// segment_match outputs frame-for-frame on real telemetry.  Reads telemetry
/// from the Anya drive and SKIPS gracefully when it is not mounted (so CI /
/// other machines pass).  Reference values are the Python detect_serve_events
/// output as of this commit — regenerate if the pipeline changes:
///   python3 -c "from pipeline.point_segmenter import *; m=load_telemetry(P); \
///     cfg=SegmenterConfig(); cfg.frame_height_px=float(m.meta['analysis_size'][1]); \
///     suppress_static_candidates(m,cfg); \
///     print([round(e.t,3) for e in detect_serve_events(m,SIDE,cfg)])"
///
/// Covers near + far events (slices 3b/3c); full segments extend this in 3d.

class _Case {
  final String path;
  final List<double> near;
  final List<double> far;
  const _Case(this.path, this.near, this.far);
}

void main() {
  const cases = [
    _Case(
      '/Volumes/Anya/Data/68/match_match_telemetry.jsonl',
      [
        8.308, 44.979, 49.667, 84.236, 103.288, 109.344, 275.897, 281.235,
        950.731, 962.327, 992.924, 1037.052, 1058.391, 1086.469, 1089.973,
        1117.868, 1150.384, 1199.868, 1206.558, 1238.256, 1242.728, 1273.392,
        1287.023,
      ],
      [
        8.292, 38.472, 44.946, 94.029, 103.238, 140.409, 159.929, 183.119,
        199.202, 224.094, 246.150, 256.794, 269.440, 308.880, 330.118, 366.772,
        419.609, 474.298, 479.804, 528.803, 565.908, 577.736, 614.874, 637.747,
        642.936, 674.668, 682.726, 693.821, 702.396, 726.804, 759.087, 773.935,
        820.699, 842.038, 852.632, 891.021, 896.577, 969.634, 992.874,
        1027.826, 1058.357, 1082.148, 1110.594, 1117.818, 1146.213, 1197.615,
        1232.501, 1264.967, 1282.702,
      ],
    ),
    _Case(
      '/Volumes/Anya/Data/23/snippet_match_telemetry.jsonl',
      [],
      [
        14.047, 40.541, 46.913, 75.542, 85.252, 112.112, 127.494, 159.359,
        172.305, 183.250, 263.196, 274.608, 287.087, 315.582, 358.625, 367.334,
        381.248, 398.531,
      ],
    ),
    _Case(
      '/Volumes/Anya/Data/21/snippet_match_telemetry.jsonl',
      [
        59.259, 220.454, 250.617, 258.925, 289.523, 305.439, 308.775, 322.422,
        339.773, 356.490, 375.675,
      ],
      [
        0.234, 30.664, 48.348, 60.394, 76.643, 223.890, 241.608, 265.999,
        290.390, 340.507,
      ],
    ),
  ];

  void expectMatch(List<double> got, List<double> expected, String label) {
    // ignore: avoid_print
    print('$label: dart ${got.length} vs python ${expected.length}');
    expect(got.length, expected.length, reason: 'event count mismatch: $label');
    for (var i = 0; i < expected.length; i++) {
      expect(got[i], closeTo(expected[i], 0.05),
          reason: 'event $i time mismatch: $label');
    }
  }

  for (final c in cases) {
    final name = c.path.split('/')[4];
    test('golden-master serve events: folder $name', () {
      final f = File(c.path);
      if (!f.existsSync()) {
        markTestSkipped('telemetry not on disk');
        return;
      }
      final m = loadTelemetry(f.readAsStringSync());
      final cfg = SegmenterConfig();
      final size = m.meta['analysis_size'] as List?;
      if (size != null) cfg.frameHeightPx = (size[1] as num).toDouble();
      suppressStaticCandidates(m, cfg);

      final near = [for (final e in detectServeEvents(m, 'near', cfg)) e.t];
      final far = [for (final e in detectServeEvents(m, 'far', cfg)) e.t];
      expectMatch(near, c.near, 'folder $name near');
      expectMatch(far, c.far, 'folder $name far');
    });
  }
}
