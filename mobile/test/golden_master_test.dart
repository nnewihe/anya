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
  final String segs; // "side:serveT:endT:start:end:method;..." from Python
  const _Case(this.path, this.near, this.far, this.segs);
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
      _segs68,
    ),
    _Case(
      '/Volumes/Anya/Data/23/snippet_match_telemetry.jsonl',
      [],
      [
        14.047, 40.541, 46.913, 75.542, 85.252, 112.112, 127.494, 159.359,
        172.305, 183.250, 263.196, 274.608, 287.087, 315.582, 358.625, 367.334,
        381.248, 398.531,
      ],
      _segs23,
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
      _segs21,
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

    test('golden-master segments: folder $name', () {
      final f = File(c.path);
      if (!f.existsSync()) {
        markTestSkipped('telemetry not on disk');
        return;
      }
      final m = loadTelemetry(f.readAsStringSync());
      final segs = segmentMatch(m);
      final expected = c.segs.split(';');
      // ignore: avoid_print
      print('folder $name segments: dart ${segs.length} vs python '
          '${expected.length}');
      expect(segs.length, expected.length,
          reason: 'segment count mismatch: $name');
      for (var i = 0; i < expected.length; i++) {
        final p = expected[i].split(':'); // side:serveT:endT:start:end:method
        final s = segs[i];
        expect(s.side, p[0], reason: 'seg $i side ($name)');
        expect(s.serveT, closeTo(double.parse(p[1]), 0.05),
            reason: 'seg $i serveT ($name)');
        expect(s.endT, closeTo(double.parse(p[2]), 0.05),
            reason: 'seg $i endT ($name)');
        expect(s.start, closeTo(double.parse(p[3]), 0.05),
            reason: 'seg $i start ($name)');
        expect(s.end, closeTo(double.parse(p[4]), 0.05),
            reason: 'seg $i end ($name)');
        expect(s.endMethod, p[5], reason: 'seg $i method ($name)');
      }
    });
  }
}

// Python segment_match references (side:serveT:endT:start:end:method), as of
// this commit.  Regenerate with:
//   python3 -c "from pipeline.point_segmenter import *; m=load_telemetry(P); \
//     print(';'.join(f'{s.side}:{s.serve_t:.2f}:{s.end_t:.2f}:{s.start:.2f}:\
//     {s.end:.2f}:{s.end_method}' for s in segment_match(m,verbose=False)))"
const _segs68 =
    'near:8.31:39.51:6.31:41.51:trace;near:44.98:54.99:42.98:56.99:trace+activity;near:84.24:88.74:82.24:90.74:trace+activity;far:94.03:98.95:89.53:100.95:trace;near:103.29:126.39:101.29:128.39:trace;far:140.41:151.55:135.91:153.55:trace;far:159.93:169.00:155.43:171.00:trace+activity;far:183.12:196.47:178.62:198.47:trace;far:199.20:213.42:194.70:215.42:trace+activity;far:224.09:235.11:219.59:237.11:trace+activity;far:246.15:252.22:241.65:254.22:trace;far:256.79:261.32:252.29:263.32:trace+activity;near:275.90:289.76:273.90:291.76:trace+activity;far:308.88:310.62:304.38:312.62:trace;far:330.12:344.60:325.62:346.60:trace+activity;far:366.77:377.43:362.27:379.43:trace;far:419.61:424.06:415.11:426.06:trace+activity;far:474.30:475.80:469.80:477.80:trace;far:528.80:533.09:524.30:535.09:trace;far:565.91:568.48:561.41:570.48:trace;far:577.74:589.93:573.24:591.93:trace+activity;far:614.87:616.48:610.37:618.48:trace;far:637.75:639.25:633.25:641.25:trace;far:674.67:677.74:670.17:679.74:trace;far:682.73:689.32:678.23:691.32:trace;far:693.82:695.52:689.32:697.52:trace;far:702.40:711.52:697.90:713.52:trace+activity;far:726.80:737.25:722.30:739.25:trace;far:759.09:769.93:754.59:771.93:trace;far:773.94:794.56:769.44:796.56:trace+activity;far:820.70:823.72:816.20:825.72:trace;far:842.04:846.99:837.54:848.99:trace+activity;far:852.63:868.73:848.13:870.73:trace+activity;far:891.02:928.01:886.52:930.01:trace;near:950.73:960.57:948.73:962.57:trace;near:962.33:982.36:960.33:984.36:trace;near:992.92:1017.67:990.92:1019.67:trace+activity;far:1027.83:1035.30:1023.33:1037.30:trace;near:1037.05:1053.50:1035.05:1055.50:trace;near:1058.39:1061.91:1056.39:1063.91:trace;near:1086.47:1094.24:1084.47:1096.24:trace+activity;near:1117.87:1131.75:1115.87:1133.75:trace+activity;near:1150.38:1168.75:1148.38:1170.75:trace;near:1199.87:1218.82:1197.87:1220.82:trace+activity;near:1238.26:1256.19:1236.26:1258.19:trace;far:1264.97:1271.34:1260.47:1273.34:trace;near:1273.39:1285.27:1271.39:1287.27:trace;near:1287.02:1292.06:1285.02:1294.06:trace+activity';
const _segs23 =
    'far:14.05:34.20:9.55:36.20:trace;far:40.54:42.58:36.04:44.58:trace;far:75.54:77.04:71.04:79.04:trace;far:85.25:86.75:80.75:88.75:trace;far:112.11:117.58:107.61:119.58:trace;far:127.49:136.64:122.99:138.64:trace;far:159.36:160.86:154.86:162.86:trace;far:172.31:178.34:167.81:180.34:trace;far:183.25:186.99:178.75:188.99:trace;far:263.20:268.13:258.70:270.13:trace+activity;far:274.61:276.11:270.11:278.11:trace+activity;far:287.09:295.40:282.59:297.40:trace+activity;far:315.58:318.38:311.08:320.38:trace;far:358.62:362.70:354.12:364.70:trace;far:367.33:369.50:362.83:371.50:trace;far:381.25:382.75:376.75:384.75:trace+activity;far:398.53:418.45:394.03:420.29:trace';
const _segs21 =
    'far:0.23:2.60:0.00:4.60:trace;far:30.66:41.84:26.16:43.84:trace;far:48.35:55.96:43.85:57.96:trace;near:59.26:74.97:57.26:76.97:trace+activity;far:76.64:86.05:72.14:88.05:trace;near:220.45:236.24:218.45:238.24:trace;far:241.61:249.05:237.11:251.05:trace;near:250.62:253.05:248.62:255.05:trace;near:258.93:273.77:256.93:275.77:trace;near:289.52:298.46:287.52:300.46:trace;near:305.44:320.19:303.44:322.19:trace;near:322.42:325.36:320.42:327.36:trace+activity;near:339.77:341.27:337.77:343.27:trace;near:356.49:359.13:354.49:361.13:trace;near:375.68:377.38:373.68:379.38:trace';
