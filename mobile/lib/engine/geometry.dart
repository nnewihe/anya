/// Even-odd ray-cast point-in-polygon. Returns true for interior points.
/// (The court homography that used to live here is gone — the engine is now
/// fully pixel-space; see active_zone.dart and telemetry.dart.)
bool pointInPolygon(double x, double y, List<List<double>> poly) {
  var inside = false;
  final n = poly.length;
  for (var i = 0, j = n - 1; i < n; j = i++) {
    final xi = poly[i][0], yi = poly[i][1];
    final xj = poly[j][0], yj = poly[j][1];
    final intersect = ((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / ((yj - yi) == 0 ? 1e-12 : (yj - yi)) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}
