// Headless macOS benchmark harness for the real on-device rally engine
// (mobile/lib/engine/engine.dart), used to measure before/after compute time
// for perf-optimization work. Mirrors analyze_ios_main.dart but targets
// macOS desktop (system ffmpeg FrameSource, no app-sandbox file restrictions)
// and exits the process on completion so it can be driven from a script.
//
// Run:  flutter build macos --release -t lib/analyze_bench_main.dart
//       build/macos/Build/Products/Release/anya_tennis.app/Contents/MacOS/anya_tennis
//
// Writes a JSON report to /tmp/anya_bench_result.json and prints a
// BENCH|... line per stage plus a final BENCH_RESULT|<json> line to stdout.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import 'engine/engine_isolate.dart';

const String _clipPath =
    '/Users/tennis/Documents/Code/Laptop/src/anya/.claude/worktrees/busy-wiles-1d4991/spikes/fixtures/clip/clip30.mp4';
const String _reportPath = '/tmp/anya_bench_result.json';
const String _label = String.fromEnvironment('BENCH_LABEL', defaultValue: 'run');

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
    print('BENCH|$s');
    setState(() => _log = '$_log\n$s');
  }

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    final report = <String, dynamic>{'label': _label};
    try {
      final clip = File(_clipPath);
      if (!clip.existsSync()) {
        throw 'clip not found: $_clipPath';
      }
      _line('clip: ${clip.path}  (${clip.lengthSync()} bytes)');

      // Runs the real shipped path: Engine loading + the whole analyze loop
      // happen inside a background worker isolate (engine_isolate.dart), same
      // as match_setup_screen.dart. Total wall time includes isolate spawn +
      // isolate-local model load, matching what the app actually experiences.
      final sw = Stopwatch()..start();
      var lastPct = -1;
      var lastStage = '';
      final stageSw = Stopwatch()..start();
      final result = await analyzeInBackground(
        videoPath: clip.path,
        writeReel: true,
        reelPath: '/tmp/anya_bench_reel_$_label.mp4',
        onProgress: (frac, msg) {
          final pct = (frac * 100).floor();
          if (pct >= lastPct + 5) {
            lastPct = pct;
            _line('  $pct%  $msg  (+${stageSw.elapsedMilliseconds}ms)');
            stageSw.reset();
          }
          if (lastStage.isEmpty) lastStage = msg;
        },
      );
      final elapsedMs = sw.elapsedMilliseconds;

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
        'analyze_ms': elapsedMs,
        'ms_per_frame': frames > 0 ? elapsedMs / frames : null,
        'segment_count': segs.length,
        'segments': segs,
        'reel': result.reelPath,
        'reel_exists':
            result.reelPath != null ? File(result.reelPath!).existsSync() : false,
        'reel_bytes': result.reelPath != null && File(result.reelPath!).existsSync()
            ? File(result.reelPath!).lengthSync()
            : 0,
        'RESULT': 'OK',
      });
      _line('segments: ${segs.length}');
      for (final s in segs) {
        _line('  ${s['start']}s - ${s['end']}s  (${s['origin']})');
      }
      _line('analyze: ${elapsedMs}ms  '
          '(${(elapsedMs / frames).toStringAsFixed(2)}ms/frame over $frames frames)');
      _line('RESULT: OK');
    } catch (e, st) {
      report['ERROR'] = '$e';
      _line('ERROR: $e');
      _line(st.toString().split('\n').take(6).join(' | '));
    }
    try {
      File(_reportPath)
          .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
      _line('wrote $_reportPath');
      // ignore: avoid_print
      print('BENCH_RESULT|${jsonEncode(report)}');
    } catch (_) {}
    exit(report['RESULT'] == 'OK' ? 0 : 1);
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Anya Bench')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Text(_log,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
          ),
        ),
      );
}
