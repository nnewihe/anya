import 'dart:math' as math;

import 'config.dart';

/// Automatic active-zone estimation — replaces the interactive 8-point
/// polygon (and the court-corner homography it implied).
///
/// The players calibrate the court. During the pre-scan (the same ~50 random
/// frames already decoded for the exclusion-zone scan) we collect every
/// confident player detection's FEET point (cx, y2) plus box height. Two
/// depth clusters naturally form: a near-court cluster (large, apparent size
/// varies a lot as the player moves toward/away from the close camera) and a
/// far-court cluster (small, size varies little since it's far from camera).
///
/// Clustering is done on BOX HEIGHT via largest-gap (1D natural-breaks, k=2)
/// — NOT on y-position. A near-court player's height varies far more (they
/// can be 3x closer at one end of the court than the other) than their
/// y-position varies relative to a hard median split, so splitting by
/// position risks slicing the near cluster in half and mislabeling real
/// near-player detections as "far" (this was tried and empirically failed —
/// see git history). Height gaps between genuine near/far clusters are large
/// and reliable because perspective compresses apparent size a lot more than
/// it compresses screen position for a court-length span.
///
/// Width estimation:
///   • Near corridor: directly from OBSERVED near-cluster feet spread (real
///     signal — near players are usually sampled widely enough across ~50
///     frames to reveal the true corridor).
///   • Far corridor: the far cluster is typically sparse (few far-player
///     detections in a random sample), so its observed spread under-covers.
///     Instead we propagate the near corridor's width by the height RATIO
///     (hFar/hNear) — a scale-free "similar triangles" assumption (width and
///     apparent height both shrink proportionally with depth for a
///     rectilinear lens) that needs no assumed real-world player height or
///     court dimension, unlike a fixed feet-per-pixel constant.
///
/// The result is a convex quad [BL, BR, TR, TL] in analysis-frame pixels,
/// directly usable by pointInPolygon. Returns null when there isn't enough
/// player evidence to fit (caller falls back to full frame).
List<List<double>>? estimateActiveZone(
  List<FeetSample> feet, {
  int minSamples = 8,
}) {
  if (feet.length < minSamples) return null;
  const w = EngineConfig.analysisWidth * 1.0;
  const h = EngineConfig.analysisHeight * 1.0;

  final split = _splitByHeightGap(feet);
  if (split == null) return null;
  final near = split.near, far = split.far;
  if (near.length < 3 || far.length < 3) return null;

  final nearH = _pct([for (final f in near) f.boxH]..sort(), 50);
  final farH = _pct([for (final f in far) f.boxH]..sort(), 50);
  if (farH <= 0 || nearH <= 1.2 * farH) return null; // not a real depth split

  // Vertical extent: min/max + height-scaled margins (players stand back
  // from the baseline; the ball flies above the far court). Every sample is
  // already a confident (conf≥playerConf) detection, not raw noise, so the
  // full observed range is trusted rather than percentile-trimmed — with
  // only ~dozens of samples per depth cluster, trimming a "5th percentile"
  // discards genuine single-sample evidence at the true court edge (see
  // git history: this under-covered the corridor on real footage).
  final nearYs = [for (final f in near) f.y];
  final farYs = [for (final f in far) f.y];
  final yBottom = math.min(h, nearYs.reduce(math.max) + 0.4 * nearH);
  final headroom = (2.2 * farH).clamp(0.08 * h, 0.35 * h);
  final yTop = math.max(0.0, farYs.reduce(math.min) - headroom);
  if (yBottom - yTop < 0.25 * h) return null;

  // Near corridor: observed spread + a margin for doubles alleys / the
  // sample not quite reaching the true sideline.
  final nearXs = [for (final f in near) f.x];
  var xNearL = math.max(0.0, nearXs.reduce(math.min) - 0.07 * w);
  var xNearR = math.min(w, nearXs.reduce(math.max) + 0.07 * w);
  if (xNearR - xNearL < 0.25 * w) return null;

  // Far corridor: propagate the near width via the height ratio (self-
  // similar perspective), centred on the far cluster's observed axis, then
  // take the wider of that estimate and the far cluster's own observed
  // spread (in case the far sample happened to be unusually wide). The far
  // cluster is typically sparse (few far-player detections per scan), so a
  // lower minimum-width floor than the near corridor is appropriate.
  final farXs = [for (final f in far) f.x]..sort();
  final axisFar = _pct(farXs, 50);
  final scaledHalfW = (xNearR - xNearL) / 2 * (farH / nearH);
  final observedHalfW = (farXs.last - farXs.first) / 2 + 0.04 * w;
  final halfWFar = math.max(scaledHalfW, observedHalfW);
  var xFarL = math.max(0.0, axisFar - halfWFar);
  var xFarR = math.min(w, axisFar + halfWFar);
  if (xFarR - xFarL < 0.05 * w) return null;

  return [
    [xNearL, yBottom], // BL
    [xNearR, yBottom], // BR
    [xFarR, yTop], // TR
    [xFarL, yTop], // TL
  ];
}

/// One player detection's ground-contact evidence: feet point + box height.
class FeetSample {
  final double x; // feet x = box centre x
  final double y; // feet y = box bottom (y2)
  final double boxH;
  const FeetSample(this.x, this.y, this.boxH);
}

class _Split {
  final List<FeetSample> near; // taller boxes
  final List<FeetSample> far; // shorter boxes
  const _Split(this.near, this.far);
}

/// 1D natural-breaks split (k=2) on box height: sort ascending, cut at the
/// single largest gap. Returns null if fewer than 2 samples.
_Split? _splitByHeightGap(List<FeetSample> feet) {
  if (feet.length < 2) return null;
  final sorted = List<FeetSample>.of(feet)..sort((a, b) => a.boxH.compareTo(b.boxH));
  var bestGap = -1.0;
  var bestIdx = 1;
  for (var i = 1; i < sorted.length; i++) {
    final gap = sorted[i].boxH - sorted[i - 1].boxH;
    if (gap > bestGap) {
      bestGap = gap;
      bestIdx = i;
    }
  }
  final far = sorted.sublist(0, bestIdx); // shorter boxes = further away
  final near = sorted.sublist(bestIdx); // taller boxes = closer
  return _Split(near, far);
}

/// Linear-interpolated percentile of a SORTED list.
double _pct(List<double> sorted, double p) {
  if (sorted.isEmpty) return 0;
  if (sorted.length == 1) return sorted.first;
  final pos = (p / 100) * (sorted.length - 1);
  final lo = pos.floor(), hi = pos.ceil();
  if (lo == hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
