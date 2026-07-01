// Spike B — prove ONNX Runtime works inside a Flutter (macOS desktop) app and
// reproduces the Python golden fixture.
//
// It loads the pre-letterboxed input tensor (input_960.f32) and both exported
// models from assets, runs inference, decodes detections (end2end [1,300,6] for
// the player model; raw [1,5,N] + threshold + NMS for the ball model),
// un-letterboxes to 960x540 frame coords, and compares to the expected boxes
// baked into *_meta.json by spikes/make_fixture.py.
//
// Results are printed with a "SPIKEB|" prefix and written to
// /tmp/onnx_spike_result.json (sandbox disabled in DebugProfile.entitlements).

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SpikeApp());
}

class SpikeApp extends StatefulWidget {
  const SpikeApp({super.key});
  @override
  State<SpikeApp> createState() => _SpikeAppState();
}

class _SpikeAppState extends State<SpikeApp> {
  String _log = 'running…';

  @override
  void initState() {
    super.initState();
    _run();
  }

  void _line(String s) {
    // ignore: avoid_print
    print('SPIKEB|$s');
    setState(() => _log = '$_log\n$s');
  }

  Future<void> _run() async {
    final report = <String, dynamic>{};
    try {
      OrtEnv.instance.init();
      _line('OrtEnv initialized, ORT ${OrtEnv.version}');

      // Shared pre-letterboxed input tensor [1,3,960,960].
      final raw = await rootBundle.load('assets/input_960.f32');
      final input = raw.buffer.asFloat32List();
      _line('loaded input_960.f32 (${input.length} floats)');

      for (final name in ['yolo26n', 'ball_best']) {
        final meta = jsonDecode(utf8.decode(
                (await rootBundle.load('assets/${name}_meta.json'))
                    .buffer
                    .asUint8List())) as Map<String, dynamic>;
        final lb = meta['letterbox'] as Map<String, dynamic>;
        final r = (lb['r'] as num).toDouble();
        final dw = (lb['dw'] as num).toDouble();
        final dh = (lb['dh'] as num).toDouble();
        final conf = (meta['conf_threshold'] as num).toDouble();
        final expected = _boxesFrom(meta['expected_boxes_manual_ort']);

        final modelBytes =
            (await rootBundle.load('assets/$name.onnx')).buffer.asUint8List();
        final opts = OrtSessionOptions()
          ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
          ..setIntraOpNumThreads(2);
        var ep = 'cpu';
        try {
          if (opts.appendCoreMLProvider(CoreMLFlags.useNone)) ep = 'coreml';
        } catch (e) {
          _line('$name: CoreML EP unavailable ($e) — using CPU');
        }
        final session = OrtSession.fromBuffer(modelBytes, opts);
        final inputName = session.inputNames.first;
        _line('$name: session ready (EP=$ep)');

        // Timed runs (first is warmup).
        List<List<double>> out = const [];
        final times = <int>[];
        for (var i = 0; i < 8; i++) {
          final t =
              OrtValueTensor.createTensorWithDataList(input, [1, 3, 960, 960]);
          final sw = Stopwatch()..start();
          final res = session.run(OrtRunOptions(), {inputName: t});
          sw.stop();
          times.add(sw.elapsedMilliseconds);
          if (i == 0) out = _flattenOutput(res.first!.value);
          t.release();
          for (final o in res) {
            o?.release();
          }
        }
        final warm = times.skip(1).toList();
        final avg = warm.reduce((a, b) => a + b) / warm.length;

        final boxes = name == 'ball_best'
            ? _decodeRaw(out, conf, r, dw, dh)
            : _decodeEnd2End(out, conf, r, dw, dh);

        final cmp = _compare(boxes, expected);
        report[name] = {
          'execution_provider': ep,
          'inference_ms_avg': avg,
          'inference_ms_min': warm.reduce(math.min),
          'n_detections': boxes.length,
          'n_expected': expected.length,
          'match': cmp,
          'boxes': boxes.map((b) => b.toJson()).toList(),
        };
        _line('$name: ${avg.toStringAsFixed(1)}ms/frame  '
            'got=${boxes.length} exp=${expected.length}  '
            '${cmp['pass'] == true ? "PASS ✅" : "FAIL ❌"} '
            '(worstIoU=${cmp['worst_iou']}, worstConfD=${cmp['worst_conf_delta']})');
        session.release();
      }

      final allPass =
          report.values.every((v) => (v['match'] as Map)['pass'] == true);
      report['RESULT'] = allPass ? 'PASS' : 'FAIL';
      _line('RESULT: ${allPass ? "PASS ✅" : "FAIL ❌"}');
    } catch (e, st) {
      report['ERROR'] = '$e';
      _line('ERROR: $e');
      _line(st.toString().split('\n').take(3).join(' | '));
    }

    try {
      File('/tmp/onnx_spike_result.json')
          .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
      _line('wrote /tmp/onnx_spike_result.json');
    } catch (e) {
      _line('could not write result file: $e');
    }
  }

  // gtbluesky's onnxruntime returns nested Lists. Normalize to List<List<double>>
  // representing the [dim1][dim2] grid (batch dim stripped).
  List<List<double>> _flattenOutput(dynamic value) {
    final batch = value as List;
    final grid = batch.first as List;
    return grid
        .map<List<double>>((row) =>
            (row as List).map<double>((v) => (v as num).toDouble()).toList())
        .toList();
  }

  List<_Box> _decodeEnd2End(
      List<List<double>> grid, double conf, double r, double dw, double dh) {
    final out = <_Box>[];
    for (final row in grid) {
      if (row.length < 6 || row[4] < conf) continue;
      out.add(_Box.unletter(
          row[0], row[1], row[2], row[3], row[4], row[5].round(), r, dw, dh));
    }
    out.sort((a, b) => b.conf.compareTo(a.conf));
    return out;
  }

  List<_Box> _decodeRaw(
      List<List<double>> grid, double conf, double r, double dw, double dh) {
    // grid: [5][N] -> rows 0..3 = cx,cy,w,h ; row 4 = class score (single class)
    final n = grid[0].length;
    final cand = <_Box>[];
    for (var j = 0; j < n; j++) {
      final s = grid[4][j];
      if (s < conf) continue;
      final cx = grid[0][j], cy = grid[1][j], w = grid[2][j], h = grid[3][j];
      cand.add(_Box.unletter(
          cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s, 0, r, dw, dh));
    }
    cand.sort((a, b) => b.conf.compareTo(a.conf));
    return _nms(cand, 0.45);
  }

  List<_Box> _nms(List<_Box> boxes, double iouThr) {
    final keep = <_Box>[];
    final rest = List<_Box>.from(boxes);
    while (rest.isNotEmpty) {
      final b = rest.removeAt(0);
      keep.add(b);
      rest.removeWhere((o) => b.iou(o) >= iouThr);
    }
    return keep;
  }

  List<_Box> _boxesFrom(dynamic list) => (list as List)
      .map((d) => _Box(
          (d['xyxy'][0] as num).toDouble(),
          (d['xyxy'][1] as num).toDouble(),
          (d['xyxy'][2] as num).toDouble(),
          (d['xyxy'][3] as num).toDouble(),
          (d['conf'] as num).toDouble(),
          d['cls'] as int))
      .toList();

  Map<String, dynamic> _compare(List<_Box> got, List<_Box> exp) {
    var worstIou = 1.0, worstConf = 0.0, matched = 0;
    final used = <int>{};
    for (final e in exp) {
      var best = 0.0, bj = -1;
      for (var j = 0; j < got.length; j++) {
        if (used.contains(j) || got[j].cls != e.cls) continue;
        final v = e.iou(got[j]);
        if (v > best) {
          best = v;
          bj = j;
        }
      }
      if (bj >= 0) {
        used.add(bj);
        matched++;
        worstIou = math.min(worstIou, best);
        worstConf = math.max(worstConf, (got[bj].conf - e.conf).abs());
      }
    }
    final pass = got.length == exp.length &&
        matched == exp.length &&
        worstIou >= 0.95 &&
        worstConf <= 0.02;
    return {
      'pass': pass,
      'matched': matched,
      'worst_iou': double.parse(worstIou.toStringAsFixed(4)),
      'worst_conf_delta': double.parse(worstConf.toStringAsFixed(4)),
    };
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('ONNX Spike B')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Text(_log,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
        ),
      );
}

class _Box {
  final double x1, y1, x2, y2, conf;
  final int cls;
  _Box(this.x1, this.y1, this.x2, this.y2, this.conf, this.cls);

  factory _Box.unletter(double x1, double y1, double x2, double y2, double conf,
      int cls, double r, double dw, double dh) {
    return _Box(
        (x1 - dw) / r, (y1 - dh) / r, (x2 - dw) / r, (y2 - dh) / r, conf, cls);
  }

  double iou(_Box o) {
    final ix1 = math.max(x1, o.x1), iy1 = math.max(y1, o.y1);
    final ix2 = math.min(x2, o.x2), iy2 = math.min(y2, o.y2);
    final iw = math.max(0.0, ix2 - ix1), ih = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;
    final ua = (x2 - x1) * (y2 - y1) + (o.x2 - o.x1) * (o.y2 - o.y1) - inter;
    return ua > 0 ? inter / ua : 0.0;
  }

  Map<String, dynamic> toJson() => {
        'cls': cls,
        'conf': double.parse(conf.toStringAsFixed(5)),
        'xyxy': [x1, y1, x2, y2]
            .map((v) => double.parse(v.toStringAsFixed(2)))
            .toList(),
      };
}
