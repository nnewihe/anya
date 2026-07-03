import 'dart:io';

import 'ball_tracker.dart';
import 'config.dart';
import 'exclusion_scan.dart';
import 'frame_source.dart';
import 'inference.dart';
import 'rally_detector.dart';
import 'reel.dart';
import 'telemetry.dart';

export 'rally_detector.dart' show RallySegment;

class EngineResult {
  final List<RallySegment> segments;
  final String? reelPath;
  final VideoInfo videoInfo;
  const EngineResult(this.segments, this.reelPath, this.videoInfo);
}

/// On-device replacement for the FastAPI/Celery backend: runs the whole
/// rally-detection pipeline locally and returns the segments (+ optional reel).
///
/// Fully self-calibrating — no court-corner homography, no interactive zone
/// selection. The pre-scan estimates the active zone from where the players
/// actually stand (see active_zone.dart) and masks static ball-like clutter
/// (see exclusion_scan.dart); near/far player classification is pixel-space.
class Engine {
  final OnnxDetector playerDetector;
  final OnnxDetector ballDetector;

  Engine._(this.playerDetector, this.ballDetector);

  static Future<Engine> load() async {
    final player = await OnnxDetector.fromAsset('assets/models/yolo26n.onnx');
    final ball = await OnnxDetector.fromAsset('assets/models/ball_best.onnx');
    return Engine._(player, ball);
  }

  /// Process-wide shared engine — models are loaded once and reused across
  /// analyses (loading two ONNX sessions is a few seconds).
  static Future<Engine>? _shared;
  static Future<Engine> shared() => _shared ??= load();

  /// Analyse [videoPath]: pre-scan (auto zones) → per-frame detection +
  /// tracking → rally segments (+ reel when [writeReel]).
  Future<EngineResult> analyze({
    required String videoPath,
    String? reelPath,
    bool writeReel = true,
    void Function(double fraction, String message)? onProgress,
  }) async {
    onProgress?.call(0.0, 'Opening video…');
    final source = await openFrameSource(videoPath);
    final info = source.info;

    // Pre-scan: ONE pass over ~50 random frames yields both the static
    // exclusion zones (ball-like clutter) and the auto-estimated active zone
    // (court occupancy from player feet, extruded for ball flight). Both are
    // cached next to the video in Python-pipeline-compatible files.
    onProgress?.call(0.0, 'Calibrating court zones…');
    final pre = await preScanVideo(
      videoPath: videoPath,
      info: info,
      ballDetector: ballDetector,
      playerDetector: playerDetector,
      onProgress: (done, total) => onProgress?.call(
          0.05 * done / total, 'Calibrating court zones ($done/$total)…'),
    );

    final telemetry = TelemetryProvider(
      playerDetector: playerDetector,
      ballDetector: ballDetector,
      activeZone: pre.activeZone,
      exclusionZones: pre.exclusionZones,
      fps: info.fps,
    );
    final ballTracker = BallTrackManager(
      info.fps,
      perspectiveScale:
          makeImageRowPerspective(EngineConfig.analysisHeight.toDouble()),
    );

    final segments = await collectRallySegments(
      frames: source.frames(),
      fps: info.fps,
      totalFrames: info.totalFrames,
      telemetry: telemetry,
      ballTracker: ballTracker,
      progressCb: (cur, total) {
        if (total > 0) {
          // 0–5% pre-scan, 5–95% analysis, 95–100% reel cut.
          onProgress?.call(
              0.05 + 0.90 * cur / total, 'Analyzing frame $cur/$total');
        }
      },
    );
    await source.dispose();

    String? outPath;
    if (writeReel && segments.isNotEmpty) {
      onProgress?.call(0.95, 'Detected ${segments.length} rallies — cutting reel…');
      outPath = reelPath ?? _defaultReelPath(videoPath);
      final ok = await createHighlightReel(
        videoPath: videoPath,
        segments: [for (final s in segments) (s.start, s.end)],
        outputPath: outPath,
      );
      if (!ok) outPath = null;
    }

    onProgress?.call(1.0, 'Done');
    return EngineResult(segments, outPath, info);
  }

  String _defaultReelPath(String videoPath) {
    final dir = File(videoPath).parent.path;
    final name = videoPath.split(Platform.pathSeparator).last;
    final stem = name.contains('.') ? name.substring(0, name.lastIndexOf('.')) : name;
    return '$dir${Platform.pathSeparator}${stem}_rallies.mp4';
  }

  void dispose() {
    playerDetector.dispose();
    ballDetector.dispose();
  }
}
