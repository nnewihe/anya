import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'active_zone.dart';
import 'config.dart';
import 'dbscan.dart';
import 'frame_source.dart';
import 'inference.dart';

/// Combined pre-scan — one pass over ~50 random frames feeding BOTH:
///
///   • Static exclusion zones (port of utilities.create_auto_exclusion_zones):
///     ball detector at very low conf; DBSCAN clusters of detection centres
///     mark ball-like clutter that sits still across random samples (baskets,
///     stray balls, court dots). Cached as `<stem>_exclusion_cache.json`, the
///     SAME path/format as the Python pipeline, so caches interop.
///
///   • Automatic active zone (replaces interactive corner/zone selection):
///     player detector feet points fit the court-occupancy trapezoid, which
///     is extruded upward to approximate the projected 3D ball-flight volume
///     (see active_zone.dart). Honors a Python-style active_zone_config.json
///     next to the video when present; otherwise estimates and caches to the
///     same file (silent no-op where the directory isn't writable).
///
/// Both detectors run on the SAME decoded frames, so adding the zone
/// estimate costs no extra video decoding.
class PreScanResult {
  final List<List<int>> exclusionZones;
  final List<List<double>> activeZone;

  /// True when the active zone came from the player-feet estimator this run
  /// (as opposed to a cache file or the full-frame fallback).
  final bool activeZoneEstimated;
  const PreScanResult(
      this.exclusionZones, this.activeZone, this.activeZoneEstimated);
}

Future<PreScanResult> preScanVideo({
  required String videoPath,
  required VideoInfo info,
  required OnnxDetector ballDetector,
  required OnnxDetector playerDetector,
  int numFrames = 50,
  void Function(int done, int total)? onProgress,
}) async {
  var exclusion = loadCachedExclusionZones(videoPath);
  var zone = loadActiveZoneFile(videoPath);
  final needBall = exclusion == null;
  final needFeet = zone == null;

  var estimated = false;
  if ((needBall || needFeet) && info.totalFrames >= numFrames && info.fps > 0) {
    final rng = math.Random();
    final pickedSet = <int>{};
    while (pickedSet.length < numFrames) {
      pickedSet.add(rng.nextInt(info.totalFrames));
    }
    // Ascending order, not pick order: each grabAnalysisFrame call seeks
    // (`-ss` before `-i`) and re-decodes from the nearest keyframe
    // independently, so sample order doesn't affect per-call decode cost —
    // but a monotonically increasing seek pattern is friendlier to the OS
    // page cache (the source file's already-read byte range keeps growing
    // forward instead of jumping backward and re-reading cold regions).
    final picked = pickedSet.toList()..sort();

    final centers = <List<double>>[];
    final feet = <FeetSample>[];
    var done = 0;
    for (final idx in picked) {
      final rgb = await grabAnalysisFrame(videoPath, idx / info.fps);
      done++;
      onProgress?.call(done, numFrames);
      if (rgb == null) continue;
      final tensor = letterboxRgbToTensor(rgb);
      if (needBall) {
        for (final b in ballDetector.detect(
            tensor, EngineConfig.exclusionScanConf,
            classIndex: EngineConfig.ballClassIndex)) {
          centers.add([b.cx, b.cy]);
        }
      }
      if (needFeet) {
        for (final p in playerDetector.detect(tensor, EngineConfig.playerConf,
            classIndex: EngineConfig.playerClassIndex)) {
          feet.add(FeetSample(p.cx, p.y2, p.y2 - p.y1));
        }
      }
    }

    if (needBall) {
      exclusion = exclusionZonesFromDetections(centers,
          eps: EngineConfig.exclusionEps,
          minSamples: EngineConfig.exclusionMinSamples,
          padding: 0);
      saveCachedExclusionZones(videoPath, exclusion);
    }
    if (needFeet) {
      zone = estimateActiveZone(feet);
      if (zone != null) {
        estimated = true;
        saveActiveZoneFile(videoPath, zone);
      }
    }
  }

  return PreScanResult(
      exclusion ?? const [], zone ?? _fullFrameZone(), estimated);
}

List<List<double>> _fullFrameZone() {
  const w = EngineConfig.analysisWidth * 1.0;
  const h = EngineConfig.analysisHeight * 1.0;
  return [
    [0.0, 0.0],
    [w, 0.0],
    [w, h],
    [0.0, h],
  ];
}

// ── Cache files (Python-pipeline-compatible paths & formats) ──────────────

String _exclusionCachePath(String videoPath) {
  final f = File(videoPath);
  final name = f.uri.pathSegments.last;
  final stem =
      name.contains('.') ? name.substring(0, name.lastIndexOf('.')) : name;
  return '${f.parent.path}${Platform.pathSeparator}${stem}_exclusion_cache.json';
}

String _activeZonePath(String videoPath) =>
    '${File(videoPath).parent.path}${Platform.pathSeparator}active_zone_config.json';

/// Load cached zones ([[x1,y1,x2,y2], ...]).
List<List<int>>? loadCachedExclusionZones(String videoPath) {
  try {
    final f = File(_exclusionCachePath(videoPath));
    if (!f.existsSync()) return null;
    final data = jsonDecode(f.readAsStringSync()) as List;
    return [
      for (final z in data)
        [for (final v in z as List) (v as num).toInt()]
    ];
  } catch (_) {
    return null; // unreadable cache → recompute
  }
}

void saveCachedExclusionZones(String videoPath, List<List<int>> zones) {
  try {
    File(_exclusionCachePath(videoPath)).writeAsStringSync(jsonEncode(zones));
  } catch (_) {
    // Video directory not writable (e.g. iOS sandbox) — skip caching.
  }
}

/// Load a Python-style active-zone polygon ([[x,y], ...], ≥3 points).
List<List<double>>? loadActiveZoneFile(String videoPath) {
  try {
    final f = File(_activeZonePath(videoPath));
    if (!f.existsSync()) return null;
    final data = jsonDecode(f.readAsStringSync()) as List;
    final poly = [
      for (final p in data)
        [((p as List)[0] as num).toDouble(), (p[1] as num).toDouble()]
    ];
    return poly.length >= 3 ? poly : null;
  } catch (_) {
    return null;
  }
}

void saveActiveZoneFile(String videoPath, List<List<double>> zone) {
  try {
    File(_activeZonePath(videoPath)).writeAsStringSync(jsonEncode([
      for (final p in zone) [p[0].round(), p[1].round()]
    ]));
  } catch (_) {
    // Not writable — skip caching.
  }
}
