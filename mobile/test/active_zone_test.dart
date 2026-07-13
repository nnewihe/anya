import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:anya_tennis/engine/active_zone.dart';
import 'package:anya_tennis/engine/geometry.dart';

// Synthetic court occupancy resembling the langmead camera geometry:
// near player roams the near baseline (y≈400-430, x 150..850, box h≈120),
// far player roams the far baseline (y≈250-270, x 400..580, box h≈45).
List<FeetSample> _syntheticFeet(int n) {
  final rng = math.Random(7);
  final feet = <FeetSample>[];
  for (var i = 0; i < n; i++) {
    feet.add(FeetSample(
        150 + rng.nextDouble() * 700, 400 + rng.nextDouble() * 30, 120));
    feet.add(FeetSample(
        400 + rng.nextDouble() * 180, 250 + rng.nextDouble() * 20, 45));
  }
  return feet;
}

void main() {
  test('estimates a sane perspective quad from player feet', () {
    final zone = estimateActiveZone(_syntheticFeet(25))!;
    expect(zone.length, 4);

    // Structure: near edge below far edge; far edge lifted above the far
    // feet (flight headroom); everything inside the frame.
    final yBottom = zone[0][1], yTop = zone[2][1];
    expect(yBottom, greaterThan(400));
    expect(yTop, lessThan(250)); // above the highest far feet
    for (final p in zone) {
      expect(p[0], inInclusiveRange(0, 960));
      expect(p[1], inInclusiveRange(0, 540));
    }

    // Mid-court rally ball is inside…
    expect(pointInPolygon(480, 330, zone), isTrue);
    // …a serve toss above the far player is inside (3D extrusion)…
    expect(pointInPolygon(480, yTop + 5, zone), isTrue);
    // …but the frame's top corner (lights / far wall) and a low far-side
    // point wide of the court (neighbouring court) are outside.
    expect(pointInPolygon(20, 20, zone), isFalse);
    expect(pointInPolygon(940, 255, zone), isFalse);
  });

  test('returns null on insufficient or degenerate evidence', () {
    expect(estimateActiveZone(const []), isNull);
    expect(
        estimateActiveZone(const [
          FeetSample(480, 400, 100),
          FeetSample(482, 401, 100),
          FeetSample(484, 402, 100),
        ]),
        isNull);
    // A tight single cluster (no near/far structure) must not fit.
    final cluster = [
      for (var i = 0; i < 30; i++)
        FeetSample(470 + (i % 5) * 2.0, 300 + (i % 7) * 1.0, 60)
    ];
    expect(estimateActiveZone(cluster), isNull);
  });
}
