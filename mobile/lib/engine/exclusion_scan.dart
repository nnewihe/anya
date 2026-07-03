import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'config.dart';
import 'dbscan.dart';
import 'frame_source.dart';
import 'inference.dart';

/// Static exclusion-zone scan — port of utilities.create_auto_exclusion_zones.
///
/// Samples [numFrames] random frames across the whole video, runs the ball
/// detector at a very low confidence, and DBSCAN-clusters the detection
/// centres. Objects that look like balls and sit in the same spot across
/// randomly-sampled frames (ball baskets, stray balls at the fence, court
/// dots) form dense clusters; a real rally ball never revisits the same pixel
/// across random samples often enough to reach [minSamples]. The cluster
/// bounding boxes become screen regions the per-frame ball detector ignores.
///
/// The result is cached next to the video as `<stem>_exclusion_cache.json` —
/// the SAME path and format the Python pipeline uses, so the two
/// implementations share caches. On sandboxed platforms where the video's
/// directory isn't writable, the cache write silently no-ops.
///
/// One deviation from Python: the scan runs the detector at the model's
/// native 960px input, not Config.BALL_IMGSZ=1920 (the exported ONNX graph
/// has a fixed input size). Static clutter is large/stationary, so the lower
/// scan resolution rarely changes which clusters form.
Future<List<List<int>>> scanStaticExclusionZones({
  required String videoPath,
  required VideoInfo info,
  required OnnxDetector ballDetector,
  int numFrames = 50,
  double conf = EngineConfig.exclusionScanConf,
  double eps = EngineConfig.exclusionEps,
  int minSamples = EngineConfig.exclusionMinSamples,
  int padding = 0,
  void Function(int done, int total)? onProgress,
}) async {
  final total = info.totalFrames;
  if (total < numFrames || info.fps <= 0) return [];

  // Random frame indices without replacement (mirrors random.sample).
  final rng = math.Random();
  final picked = <int>{};
  while (picked.length < numFrames) {
    picked.add(rng.nextInt(total));
  }

  final centers = <List<double>>[];
  var done = 0;
  for (final idx in picked) {
    final rgb = await grabAnalysisFrame(videoPath, idx / info.fps);
    done++;
    onProgress?.call(done, numFrames);
    if (rgb == null) continue;
    final tensor = letterboxRgbToTensor(rgb);
    for (final b in ballDetector.detect(tensor, conf,
        classIndex: EngineConfig.ballClassIndex)) {
      centers.add([b.cx, b.cy]);
    }
  }

  return exclusionZonesFromDetections(centers,
      eps: eps, minSamples: minSamples, padding: padding);
}

String _cachePath(String videoPath) {
  final f = File(videoPath);
  final name = f.uri.pathSegments.last;
  final stem = name.contains('.') ? name.substring(0, name.lastIndexOf('.')) : name;
  return '${f.parent.path}${Platform.pathSeparator}${stem}_exclusion_cache.json';
}

/// Load cached zones (Python-compatible format: [[x1,y1,x2,y2], ...]).
List<List<int>>? loadCachedExclusionZones(String videoPath) {
  try {
    final f = File(_cachePath(videoPath));
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
    File(_cachePath(videoPath)).writeAsStringSync(jsonEncode(zones));
  } catch (_) {
    // Video directory not writable (e.g. iOS sandbox) — skip caching.
  }
}

/// Cache-or-scan wrapper mirroring AnyaTelemetryProvider.__init__'s flow.
Future<List<List<int>>> getOrScanExclusionZones({
  required String videoPath,
  required VideoInfo info,
  required OnnxDetector ballDetector,
  void Function(int done, int total)? onProgress,
}) async {
  final cached = loadCachedExclusionZones(videoPath);
  if (cached != null) return cached;
  final zones = await scanStaticExclusionZones(
    videoPath: videoPath,
    info: info,
    ballDetector: ballDetector,
    onProgress: onProgress,
  );
  saveCachedExclusionZones(videoPath, zones);
  return zones;
}
