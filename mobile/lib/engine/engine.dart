import 'dart:convert';
import 'dart:io';

import 'ball_tracker.dart';
import 'config.dart';
import 'exclusion_scan.dart';
import 'frame_source.dart';
import 'geometry.dart';
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

  /// Analyse [videoPath]. [corners] are the 4 court corners in analysis-frame
  /// (960×540) pixel space, ordered [BL, BR, TR, TL] (matching the Python
  /// homography). Returns detected segments and, if [writeReel], a reel path.
  Future<EngineResult> analyze({
    required String videoPath,
    required List<List<double>> corners,
    String? reelPath,
    bool writeReel = true,
    void Function(double fraction, String message)? onProgress,
  }) async {
    onProgress?.call(0.0, 'Opening video…');
    final source = await openFrameSource(videoPath);
    final info = source.info;

    // Homography: court corners (px) → world rectangle (ft).
    final homography = Homography.from4(corners, const [
      [0, 0],
      [EngineConfig.courtWidthFt, 0],
      [EngineConfig.courtWidthFt, EngineConfig.courtLengthFt],
      [0, EngineConfig.courtLengthFt],
    ]);

    // Active zone: honor a Python-pipeline-style active_zone_config.json next
    // to the video when present (8-point polygon in 960×540 space); otherwise
    // default to the full analysis frame (matches the headless backend).
    final activeZone = _loadActiveZone(videoPath);

    // Static exclusion zones: one-time DBSCAN scan over ~50 random frames to
    // find ball-like clutter (baskets, stray balls) and mask those regions
    // out of per-frame ball detection. Cached next to the video in the same
    // format as the Python pipeline, so the two share caches.
    onProgress?.call(0.0, 'Scanning for ball baskets / stray balls…');
    final exclusionZones = await getOrScanExclusionZones(
      videoPath: videoPath,
      info: info,
      ballDetector: ballDetector,
      onProgress: (done, total) => onProgress?.call(
          0.05 * done / total, 'Scanning for static clutter ($done/$total)…'),
    );

    final telemetry = TelemetryProvider(
      playerDetector: playerDetector,
      ballDetector: ballDetector,
      homography: homography,
      activeZone: activeZone,
      exclusionZones: exclusionZones,
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
          // 0–5% scan, 5–95% analysis, 95–100% reel cut.
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

  /// Load `<videoDir>/active_zone_config.json` (same file the Python pipeline
  /// reads/writes) or fall back to the full analysis frame.
  static List<List<double>> _loadActiveZone(String videoPath) {
    try {
      final f = File(
          '${File(videoPath).parent.path}${Platform.pathSeparator}active_zone_config.json');
      if (f.existsSync()) {
        final data = jsonDecode(f.readAsStringSync()) as List;
        final poly = [
          for (final p in data)
            [((p as List)[0] as num).toDouble(), (p[1] as num).toDouble()]
        ];
        if (poly.length >= 3) return poly;
      }
    } catch (_) {}
    const w = EngineConfig.analysisWidth * 1.0;
    const h = EngineConfig.analysisHeight * 1.0;
    return [
      [0.0, 0.0],
      [w, 0.0],
      [w, h],
      [0.0, h],
    ];
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
