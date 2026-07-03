// Headless iOS validation of the full on-device pipeline, exercising the real
// mobile ffmpeg_kit backend (FfmpegKitFrameSource / reelCutMobile /
// referenceFrameMobile via lib/engine/ffmpeg_mobile.dart) instead of the
// desktop system-ffmpeg path. Mirrors the macOS spike harness
// (spikes/onnx_dart/lib/analyze_main.dart) so results are directly comparable.
//
// Expects clip30.mp4 already placed in the app's Documents directory (see
// `xcrun simctl get_app_container ... data` + cp).
//
// Run:  flutter run -d "iPhone 16 Pro" -t lib/analyze_ios_main.dart

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'engine/engine.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _App());
}

class _App extends StatefulWidget {
  const _App();
  @override
  State<_App> createState() => _S();
}

class _S extends State<_App> {
  String _log = 'running…';
  void _line(String s) {
    // ignore: avoid_print
    print('ANALYZE_IOS|$s');
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
      final docs = await getApplicationDocumentsDirectory();
      final clip = File('${docs.path}/clip30.mp4');
      if (!clip.existsSync()) {
        throw 'clip30.mp4 not found in Documents (${docs.path})';
      }
      _line('clip: ${clip.path}  (${clip.lengthSync()} bytes)');

      final sw = Stopwatch()..start();
      final engine = await Engine.shared();
      _line('engine loaded (${sw.elapsedMilliseconds}ms)');

      sw.reset();
      var lastPct = -1;
      final result = await engine.analyze(
        videoPath: clip.path,
        writeReel: true,
        reelPath: '${docs.path}/clip30_rallies.mp4',
        onProgress: (frac, msg) {
          final pct = (frac * 100).floor();
          if (pct >= lastPct + 10) {
            lastPct = pct;
            _line('  $pct%  $msg');
          }
        },
      );
      final elapsed = sw.elapsedMilliseconds;

      final segs = [
        for (final s in result.segments)
          {
            'start': double.parse(s.start.toStringAsFixed(2)),
            'end': double.parse(s.end.toStringAsFixed(2)),
            'origin': s.origin,
          }
      ];
      final frames = result.videoInfo.totalFrames;
      report.addAll({
        'clip': clip.path,
        'fps': result.videoInfo.fps,
        'frames': frames,
        'analyze_ms': elapsed,
        'segment_count': segs.length,
        'segments': segs,
        'reel': result.reelPath,
        'reel_exists': result.reelPath != null
            ? File(result.reelPath!).existsSync()
            : false,
        'reel_bytes': result.reelPath != null && File(result.reelPath!).existsSync()
            ? File(result.reelPath!).lengthSync()
            : 0,
        'RESULT': 'OK',
      });
      _line('segments: ${segs.length}');
      for (final s in segs) {
        _line('  ${s['start']}s - ${s['end']}s  (${s['origin']})');
      }
      _line('analyze: ${elapsed}ms  reel=${result.reelPath}  '
          'exists=${report['reel_exists']}  bytes=${report['reel_bytes']}');
      _line('RESULT: OK');
    } catch (e, st) {
      report['ERROR'] = '$e';
      _line('ERROR: $e');
      _line(st.toString().split('\n').take(4).join(' | '));
    }
    try {
      final docs = await getApplicationDocumentsDirectory();
      File('${docs.path}/analyze_ios_result.json')
          .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
      _line('wrote analyze_ios_result.json');
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('iOS Engine Analyze')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Text(_log,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
          ),
        ),
      );
}
