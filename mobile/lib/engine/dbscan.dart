import 'dart:math' as math;

/// DBSCAN over 2D points, matching scikit-learn's semantics for the parameters
/// the pipeline uses (euclidean metric, `eps`, `min_samples`). Returns cluster
/// labels; -1 is noise. Used to find static ball-like false positives.
List<int> dbscan(List<List<double>> points, double eps, int minSamples) {
  final n = points.length;
  final labels = List<int>.filled(n, -2); // -2 = unvisited
  final eps2 = eps * eps;

  List<int> region(int p) {
    final out = <int>[];
    for (var q = 0; q < n; q++) {
      final dx = points[p][0] - points[q][0];
      final dy = points[p][1] - points[q][1];
      if (dx * dx + dy * dy <= eps2) out.add(q);
    }
    return out;
  }

  var cluster = -1;
  for (var p = 0; p < n; p++) {
    if (labels[p] != -2) continue;
    final neighbors = region(p);
    // sklearn counts the point itself; core point iff |neighbors| >= min_samples
    if (neighbors.length < minSamples) {
      labels[p] = -1; // noise (may be claimed by a cluster later)
      continue;
    }
    cluster += 1;
    labels[p] = cluster;
    final seeds = List<int>.of(neighbors);
    for (var i = 0; i < seeds.length; i++) {
      final q = seeds[i];
      if (labels[q] == -1) labels[q] = cluster; // border point
      if (labels[q] != -2) continue;
      labels[q] = cluster;
      final qn = region(q);
      if (qn.length >= minSamples) seeds.addAll(qn);
    }
  }
  return labels;
}

/// Bounding-box exclusion zones from clustered detection centres (mirrors
/// create_auto_exclusion_zones / get_exclusion_zones_from_frames).
List<List<int>> exclusionZonesFromDetections(
  List<List<double>> centers, {
  double eps = 12,
  int minSamples = 15,
  int padding = 0,
}) {
  if (centers.length < minSamples) return [];
  final labels = dbscan(centers, eps, minSamples);
  final byCluster = <int, List<List<double>>>{};
  for (var i = 0; i < labels.length; i++) {
    if (labels[i] < 0) continue;
    byCluster.putIfAbsent(labels[i], () => []).add(centers[i]);
  }
  final zones = <List<int>>[];
  for (final pts in byCluster.values) {
    var xMin = double.infinity, yMin = double.infinity;
    var xMax = -double.infinity, yMax = -double.infinity;
    for (final p in pts) {
      xMin = math.min(xMin, p[0]);
      yMin = math.min(yMin, p[1]);
      xMax = math.max(xMax, p[0]);
      yMax = math.max(yMax, p[1]);
    }
    zones.add([
      (xMin - padding).toInt(),
      (yMin - padding).toInt(),
      (xMax + padding).toInt(),
      (yMax + padding).toInt(),
    ]);
  }
  return zones;
}

bool isInExclusionZone(double x, double y, List<List<int>> zones) {
  for (final z in zones) {
    if (z[0] <= x && x <= z[2] && z[1] <= y && y <= z[3]) return true;
  }
  return false;
}
