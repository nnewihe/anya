// Spike A (capstone) — full on-device chain, no server:
//   real mp4  ->  ffmpeg decode to 960x540 rgb24 frames (streamed into Dart)
//             ->  letterbox to 960x960 float tensor (in Dart)
//             ->  ONNX ball detection (CoreML)  ->  detections
//
// Proves Spike A (frames into Dart) AND that it composes with Spike B, and
// measures end-to-end throughput (decode + preprocess + inference).
//
// Desktop decode here uses Process.start(ffmpeg …) as the FrameSource. On mobile
// the same rgb24 frames come from ffmpeg_kit or a native AVAssetReader/MediaCodec
// plugin behind the identical FrameSource interface — decode is not the
// bottleneck (see spikes benchmark), inference is.
//
// Run:  flutter run -d macos --release -t lib/decode_main.dart

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

const int fw = 960, fh = 540; // analysis frame
const int sz = 960; // model input (letterboxed square)
const int frameBytes = fw * fh * 3;
const int maxFrames = 300;

const _videoCandidates = [
  '/Users/tennis/Documents/Code/Laptop/src/anya/archive/out.mp4',
  '/Users/tennis/Documents/Code/Laptop/src/anya/archive/farside_serve_viz.mp4',
];

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DecodeSpikeApp());
}

class DecodeSpikeApp extends StatefulWidget {
  const DecodeSpikeApp({super.key});
  @override
  State<DecodeSpikeApp> createState() => _S();
}

class _S extends State<DecodeSpikeApp> {
  String _log = 'running…';
  void _line(String s) {
    // ignore: avoid_print
    print('SPIKEA|$s');
    setState(() => _log = '$_log\n$s');
  }

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    final report = <String, dynamic>{};
    try {
      final video = _videoCandidates.firstWhere(
          (p) => File(p).existsSync(),
          orElse: () => '');
      if (video.isEmpty) throw 'no test video found';
      _line('video: $video');

      OrtEnv.instance.init();
      final opts = OrtSessionOptions()
        ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
        ..setIntraOpNumThreads(2);
      var ep = 'cpu';
      if (opts.appendCoreMLProvider(CoreMLFlags.useNone)) ep = 'coreml';
      final bytes =
          (await rootBundle.load('assets/ball_best.onnx')).buffer.asUint8List();
      final session = OrtSession.fromBuffer(bytes, opts);
      final inputName = session.inputNames.first;
      _line('ball_best session ready (EP=$ep)');

      // Letterbox params for 960x540 -> 960x960 (r=1, pad top/bottom 210).
      const r = 1.0, dw = 0.0, dh = 210.0;

      final proc = await Process.start('ffmpeg', [
        '-v', 'error', '-i', video,
        '-vf', 'scale=$fw:$fh', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'
      ]);
      proc.stderr.transform(utf8.decoder).listen((e) {
        if (e.trim().isNotEmpty) _line('ffmpeg: ${e.trim()}');
      });

      final acc = BytesBuilder();
      var frames = 0, framesWithBall = 0;
      var decodeMs = 0, inferMs = 0;
      final swAll = Stopwatch()..start();

      await for (final chunk in proc.stdout) {
        acc.add(chunk);
        while (acc.length >= frameBytes && frames < maxFrames) {
          final all = acc.takeBytes();
          final frame = all.sublist(0, frameBytes);
          if (all.length > frameBytes) acc.add(all.sublist(frameBytes));

          final swP = Stopwatch()..start();
          final input = _letterboxToTensor(frame);
          decodeMs += swP.elapsedMilliseconds;

          final t = OrtValueTensor.createTensorWithDataList(input, [1, 3, sz, sz]);
          final swI = Stopwatch()..start();
          final res = session.run(OrtRunOptions(), {inputName: t});
          inferMs += swI.elapsedMilliseconds;

          final dets = _decodeBall(_grid(res.first!.value), 0.05, r, dw, dh);
          if (dets.isNotEmpty) framesWithBall++;
          t.release();
          for (final o in res) {
            o?.release();
          }
          frames++;
        }
        if (frames >= maxFrames) break;
      }
      proc.kill();
      swAll.stop();

      final fps = frames * 1000.0 / swAll.elapsedMilliseconds;
      report.addAll({
        'video': video,
        'execution_provider': ep,
        'frames_processed': frames,
        'frames_with_ball_detection': framesWithBall,
        'wall_ms_total': swAll.elapsedMilliseconds,
        'preprocess_ms_total': decodeMs,
        'inference_ms_total': inferMs,
        'end_to_end_fps': double.parse(fps.toStringAsFixed(1)),
        'RESULT': frames == maxFrames ? 'PASS' : 'PARTIAL',
      });
      _line('processed $frames frames in ${swAll.elapsedMilliseconds}ms  '
          '=> ${fps.toStringAsFixed(1)} fps end-to-end  '
          '(preprocess ${decodeMs}ms, infer ${inferMs}ms)  '
          'ball in $framesWithBall/$frames frames');
      _line('RESULT: ${report['RESULT']} ✅');
      session.release();
    } catch (e, st) {
      report['ERROR'] = '$e';
      _line('ERROR: $e');
      _line(st.toString().split('\n').take(3).join(' | '));
    }
    try {
      File('/tmp/decode_spike_result.json')
          .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
      _line('wrote /tmp/decode_spike_result.json');
    } catch (_) {}
  }

  Float32List _letterboxToTensor(Uint8List rgb) {
    // rgb: 960x540 row-major rgb24. Build [1,3,960,960] float, gray(114) pad.
    final out = Float32List(3 * sz * sz);
    const pad = 114 / 255.0;
    for (var i = 0; i < out.length; i++) {
      out[i] = pad; // fill; content rows overwrite below
    }
    const dhi = 210;
    for (var y = 0; y < fh; y++) {
      final ty = y + dhi;
      for (var x = 0; x < fw; x++) {
        final p = (y * fw + x) * 3;
        final rC = rgb[p] / 255.0, gC = rgb[p + 1] / 255.0, bC = rgb[p + 2] / 255.0;
        final base = ty * sz + x;
        out[base] = rC; // channel 0
        out[sz * sz + base] = gC; // channel 1
        out[2 * sz * sz + base] = bC; // channel 2
      }
    }
    return out;
  }

  List<List<double>> _grid(dynamic value) {
    final grid = (value as List).first as List;
    return grid
        .map<List<double>>(
            (row) => (row as List).map<double>((v) => (v as num).toDouble()).toList())
        .toList();
  }

  // Ball model is now exported with NMS baked in (conf=0.001) -> compact
  // [300,6] rows of x1,y1,x2,y2,conf,cls. Threshold per call-site in Dart.
  List<List<double>> _decodeBall(
      List<List<double>> grid, double conf, double r, double dw, double dh) {
    final boxes = <List<double>>[];
    for (final row in grid) {
      if (row.length < 6 || row[4] < conf) continue;
      boxes.add([(row[0] - dw) / r, (row[1] - dh) / r,
        (row[2] - dw) / r, (row[3] - dh) / r, row[4]]);
    }
    return boxes;
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Decode Spike A')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Text(_log,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
        ),
      );
}
