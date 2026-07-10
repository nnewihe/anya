import 'dart:io';

import 'config.dart';
import 'exclusion_scan.dart';
import 'frame_source.dart';
import 'geometry.dart';
import 'inference.dart';
import 'match_telemetry.dart';
import 'point_segmenter.dart';
import 'reel.dart';

/// Result of a dead-time cut.
class DeadTimeResult {
  final List<PointSegment> segments;
  final String? reelPath;
  final VideoInfo videoInfo;
  const DeadTimeResult(this.segments, this.reelPath, this.videoInfo);
}

/// On-device dead-time cutter: the mobile counterpart of the Python
/// deadtime_cutter.cut_dead_time / the desktop worker.  Runs the ported
/// stage-1 telemetry extraction (match_telemetry.dart) then stage-2
/// segmentation (point_segmenter.dart), and cuts the kept segments.
///
/// Distinct from the rally [Engine]: it needs the court-corner homography
/// (ready-band + far-crop geometry), so [analyze] takes the four calibrated
/// corners.  Far-region ball detection needs native-res crops
/// ([nativeCropProvider], Phase 4b); without one it falls back to
/// [NoFarBalls] — far serves that depend on the native crop won't form a
/// trace, exactly as the folder-68 fballs-strip test showed.
class DeadTimeEngine {
  final OnnxDetector playerDetector;
  final OnnxDetector ballDetector;

  DeadTimeEngine._(this.playerDetector, this.ballDetector);

  static Future<DeadTimeEngine> load() async {
    final player = await OnnxDetector.fromAsset('assets/models/yolo26n.onnx');
    final ball = await OnnxDetector.fromAsset('assets/models/ball_best.onnx');
    return DeadTimeEngine._(player, ball);
  }

  static Future<DeadTimeEngine>? _shared;
  static Future<DeadTimeEngine> shared() => _shared ??= load();

  /// Analyse [videoPath] with the calibrated court [corners] (BL, BR, TR, TL
  /// in analysis-frame pixels) and cut the dead time.
  Future<DeadTimeResult> analyze({
    required String videoPath,
    required List<List<double>> corners,
    NativeCropProvider? nativeCropProvider,
    String? reelPath,
    bool writeReel = true,
    void Function(double fraction, String message)? onProgress,
  }) async {
    onProgress?.call(0.0, 'Opening video…');
    final source = await openFrameSource(videoPath);
    final info = source.info;

    // Pre-scan: exclusion zones (+ active zone, though the cutter's ball
    // gating leans on the active zone less than the rally engine).
    onProgress?.call(0.0, 'Calibrating court zones…');
    final pre = await preScanVideo(
      videoPath: videoPath,
      info: info,
      ballDetector: ballDetector,
      playerDetector: playerDetector,
      onProgress: (done, total) => onProgress?.call(
          0.05 * done / total, 'Calibrating court zones ($done/$total)…'),
    );

    // Pixel→world homography from the calibrated corners.
    const w = CutterConfig.courtWidthFt, l = CutterConfig.courtLengthFt;
    final homography = Homography.from4(corners, [
      [0.0, 0.0], // BL
      [w, 0.0], // BR
      [w, l], // TR
      [0.0, l], // TL
    ]);

    final farBallSource = nativeCropProvider == null
        ? const NoFarBalls()
        : FixedFarCropSource(
            farCourtCropRect(corners), nativeCropProvider, ballDetector);

    final extractor = MatchTelemetryExtractor(
      playerDetector: playerDetector,
      ballDetector: ballDetector,
      homography: homography,
      activeZone: pre.activeZone,
      exclusionZones: pre.exclusionZones,
      fps: info.fps,
      farBallSource: farBallSource,
    );

    // Stage 1: per-frame telemetry → JSONL (cached next to the video, like
    // the Python pipeline, so re-runs skip perception).
    onProgress?.call(0.05, 'Analyzing…');
    final lines = <String>[
      MatchTelemetryExtractor.encodeMeta(MatchTelemetryExtractor.metaHeader(
          info.fps, info.totalFrames,
          hasFarBalls: nativeCropProvider != null)),
    ];
    var frameId = 0;
    await for (final rgb in source.frames()) {
      frameId += 1;
      final tSec = frameId / info.fps;
      final rec = await extractor.processFrame(rgb, tSec, frameId);
      lines.add(MatchTelemetryExtractor.encodeRecord(rec));
      if (info.totalFrames > 0 && frameId % 30 == 0) {
        onProgress?.call(0.05 + 0.85 * frameId / info.totalFrames,
            'Analyzing frame $frameId/${info.totalFrames}');
      }
    }
    await source.dispose();

    final jsonl = lines.join('\n');
    try {
      File(_telemetryPath(videoPath)).writeAsStringSync(jsonl);
    } catch (_) {
      // caching is best-effort; segmentation runs off the in-memory string.
    }

    // Stage 2: telemetry → point segments (pure Dart, golden-mastered).
    onProgress?.call(0.92, 'Detecting points…');
    final match = loadTelemetry(jsonl);
    final segments = segmentMatch(match);

    // Stage 3: cut the kept segments.  Pre-roll is baked into each segment;
    // merge_gap 1.0 keeps fault→second-serve one continuous cut.
    String? outPath;
    if (writeReel && segments.isNotEmpty) {
      onProgress?.call(
          0.95, 'Cutting ${segments.length} points…');
      outPath = reelPath ?? _defaultOutputPath(videoPath);
      final ok = await createHighlightReel(
        videoPath: videoPath,
        segments: [for (final s in segments) (s.start, s.end)],
        outputPath: outPath,
        preRoll: 0.0,
        mergeGapSec: 1.0,
      );
      if (!ok) outPath = null;
    }

    onProgress?.call(1.0, 'Done');
    return DeadTimeResult(segments, outPath, info);
  }

  String _telemetryPath(String videoPath) =>
      '${_stem(videoPath)}_match_telemetry.jsonl';

  String _defaultOutputPath(String videoPath) =>
      '${_stem(videoPath)}_no_deadtime.mp4';

  String _stem(String videoPath) {
    final dot = videoPath.lastIndexOf('.');
    return dot > 0 ? videoPath.substring(0, dot) : videoPath;
  }

  void dispose() {
    playerDetector.dispose();
    ballDetector.dispose();
  }
}
