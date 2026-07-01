import 'linalg.dart';

/// 3×3 homography and the two OpenCV operations the pipeline uses:
/// `getPerspectiveTransform` (exact 4-point) and `perspectiveTransform`.
class Homography {
  final List<double> h; // 9 elements, row-major, h[8] == 1

  Homography(this.h);

  /// Exact homography mapping the 4 `src` points to the 4 `dst` points.
  /// Mirrors cv2.findHomography on 4 correspondences (== getPerspectiveTransform).
  factory Homography.from4(List<List<double>> src, List<List<double>> dst) {
    final a = Mat(8, 8);
    final b = Mat(8, 1);
    for (var i = 0; i < 4; i++) {
      final x = src[i][0], y = src[i][1];
      final u = dst[i][0], v = dst[i][1];
      final r0 = 2 * i, r1 = 2 * i + 1;
      // u = h0 x + h1 y + h2 - h6 x u - h7 y u
      a.set(r0, 0, x);
      a.set(r0, 1, y);
      a.set(r0, 2, 1);
      a.set(r0, 6, -x * u);
      a.set(r0, 7, -y * u);
      b.d[r0] = u;
      // v = h3 x + h4 y + h5 - h6 x v - h7 y v
      a.set(r1, 3, x);
      a.set(r1, 4, y);
      a.set(r1, 5, 1);
      a.set(r1, 6, -x * v);
      a.set(r1, 7, -y * v);
      b.d[r1] = v;
    }
    final sol = a.inverse().matmul(b); // 8×1
    return Homography([...sol.d, 1.0]);
  }

  /// Map a pixel point to world coordinates.
  List<double> transform(double px, double py) {
    final w = h[6] * px + h[7] * py + h[8];
    final wx = (h[0] * px + h[1] * py + h[2]) / w;
    final wy = (h[3] * px + h[4] * py + h[5]) / w;
    return [wx, wy];
  }
}

/// Even-odd ray-cast point-in-polygon. Returns true for interior points
/// (edge cases are immaterial: the default active zone is the full frame).
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
