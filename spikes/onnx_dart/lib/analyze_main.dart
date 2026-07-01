// Phase 5 — full on-device engine end-to-end on macOS.
//   video file -> decode -> player+ball ONNX -> IMM tracker -> rally state
//   machine + Viterbi HMM -> segments (+ reel). No server.
//
// Run: flutter run -d macos --release -t lib/analyze_main.dart

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import 'engine/engine.dart';

const _clipCandidates = [
  '/Users/tennis/Documents/Code/Laptop/src/anya/.claude/worktrees/zealous-lichterman-f21da6/spikes/fixtures/clip/clip30.mp4',
  '/Users/tennis/Documents/Code/Laptop/src/anya/.claude/worktrees/zealous-lichterman-f21da6/spikes/fixtures/clip/clip90.mp4',
];

// Court corners in 960×540 analysis space, ordered [BL, BR, TR, TL].
const _corners = <List<double>>[
  [120, 510],
  [840, 510],
  [600, 140],
  [360, 140],
];

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
    print('ANALYZE|$s');
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
      final clip = _clipCandidates.firstWhere((p) => File(p).existsSync(),
          orElse: () => '');
      if (clip.isEmpty) throw 'no clip found';
      _line('clip: $clip');

      final sw = Stopwatch()..start();
      final engine = await Engine.load();
      _line('engine loaded (${sw.elapsedMilliseconds}ms)');

      sw.reset();
      var lastPct = -1;
      final result = await engine.analyze(
        videoPath: clip,
        corners: _corners,
        writeReel: true,
        reelPath: '/tmp/analyze_reel.mp4',
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
        'clip': clip,
        'fps': result.videoInfo.fps,
        'frames': frames,
        'analyze_ms': elapsed,
        'fps_processed':
            double.parse((frames * 1000.0 / elapsed).toStringAsFixed(1)),
        'segment_count': segs.length,
        'segments': segs,
        'reel': result.reelPath,
        'RESULT': 'OK',
      });
      _line('segments: ${segs.length}');
      for (final s in segs) {
        _line('  ${s['start']}s – ${s['end']}s  (${s['origin']})');
      }
      _line('analyze: ${elapsed}ms  '
          '(${(frames * 1000.0 / elapsed).toStringAsFixed(1)} fps)  reel=${result.reelPath}');
      _line('RESULT: OK ✅');
      engine.dispose();
    } catch (e, st) {
      report['ERROR'] = '$e';
      _line('ERROR: $e');
      _line(st.toString().split('\n').take(4).join(' | '));
    }
    try {
      File('/tmp/analyze_result.json')
          .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
      _line('wrote /tmp/analyze_result.json');
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Engine Analyze')),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Text(_log,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
          ),
        ),
      );
}
