import 'package:flutter_test/flutter_test.dart';
import 'package:rally_predictor/engine/geometry.dart';

/// The resurrected court homography (geometry.dart) — the cutter's near/far
/// ready-band depends on it.  Validates the two OpenCV operations the Python
/// pipeline uses: exact 4-point solve + point transform.
void main() {
  test('identity: src == dst maps points to themselves', () {
    final square = [
      [0.0, 0.0],
      [10.0, 0.0],
      [10.0, 10.0],
      [0.0, 10.0],
    ];
    final h = Homography.from4(square, square);
    final p = h.transform(3.7, 6.2);
    expect(p[0], closeTo(3.7, 1e-6));
    expect(p[1], closeTo(6.2, 1e-6));
  });

  test('pure scale: 10x10 square -> 100x100 square', () {
    final src = [
      [0.0, 0.0],
      [10.0, 0.0],
      [10.0, 10.0],
      [0.0, 10.0],
    ];
    final dst = [
      [0.0, 0.0],
      [100.0, 0.0],
      [100.0, 100.0],
      [0.0, 100.0],
    ];
    final h = Homography.from4(src, dst);
    final mid = h.transform(5.0, 5.0);
    expect(mid[0], closeTo(50.0, 1e-6));
    expect(mid[1], closeTo(50.0, 1e-6));
  });

  test('4-point exactness: each court corner maps to its world point', () {
    // A trapezoid (perspective) of pixel corners BL, BR, TR, TL → the court
    // rectangle in feet, exactly how _compute_homography builds it.
    final src = [
      [62.0, 414.0], // BL
      [887.0, 424.0], // BR
      [561.0, 270.0], // TR
      [419.0, 273.0], // TL
    ];
    final dst = [
      [0.0, 0.0],
      [27.0, 0.0],
      [27.0, 78.0],
      [0.0, 78.0],
    ];
    final h = Homography.from4(src, dst);
    for (var i = 0; i < 4; i++) {
      final w = h.transform(src[i][0], src[i][1]);
      expect(w[0], closeTo(dst[i][0], 1e-4), reason: 'corner $i wx');
      expect(w[1], closeTo(dst[i][1], 1e-4), reason: 'corner $i wy');
    }
    // A perspective map is non-affine: the pixel centroid does NOT map to the
    // world centroid (13.5, 39) — just assert the transform stays finite and
    // inside a sane range, proving the solve produced a usable matrix.
    const cx = (62 + 887 + 561 + 419) / 4.0;
    const cy = (414 + 424 + 270 + 273) / 4.0;
    final c = h.transform(cx, cy);
    expect(c[0], inInclusiveRange(-5.0, 32.0));
    expect(c[1], inInclusiveRange(-5.0, 83.0));
  });
}
