import 'dart:io';

import 'ball_tracker.dart';
import 'config.dart';
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
    final source = await FfmpegFrameSource.open(videoPath);
    final info = source.info;

    // Homography: court corners (px) → world rectangle (ft).
    final homography = Homography.from4(corners, const [
      [0, 0],
      [EngineConfig.courtWidthFt, 0],
      [EngineConfig.courtWidthFt, EngineConfig.courtLengthFt],
      [0, EngineConfig.courtLengthFt],
    ]);

    // Default active zone = full analysis frame (matches the headless backend).
    const w = EngineConfig.analysisWidth, h = EngineConfig.analysisHeight;
    final activeZone = <List<double>>[
      [0, 0],
      [w.toDouble(), 0],
      [w.toDouble(), h.toDouble()],
      [0, h.toDouble()],
    ];

    final telemetry = TelemetryProvider(
      playerDetector: playerDetector,
      ballDetector: ballDetector,
      homography: homography,
      activeZone: activeZone,
      exclusionZones: const [], // static-exclusion pre-scan is a follow-up
      fps: info.fps,
    );
    final ballTracker = BallTrackManager(
      info.fps,
      perspectiveScale: makeImageRowPerspective(h.toDouble()),
    );

    final segments = await collectRallySegments(
      frames: source.frames(),
      fps: info.fps,
      totalFrames: info.totalFrames,
      telemetry: telemetry,
      ballTracker: ballTracker,
      progressCb: (cur, total) {
        if (total > 0) {
          onProgress?.call(
              0.95 * cur / total, 'Analyzing frame $cur/$total');
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
