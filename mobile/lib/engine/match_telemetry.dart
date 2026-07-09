import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'ball_tracker.dart' show Detection;
import 'config.dart';
import 'dbscan.dart' show isInExclusionZone;
import 'geometry.dart';
import 'inference.dart';

/// Stage-1 of the dead-time cutter, ported from pipeline/match_telemetry.py.
///
/// One perception pass over the video producing a per-frame telemetry record
/// that stage 2 (point_segmenter, ported separately) turns into point
/// segments.  Emits the SAME JSONL schema as the Python extractor so the two
/// can be cross-checked frame-for-frame (the golden-master test): run Python
/// stage 1 → JSONL, run this → JSONL, diff.
///
/// Unlike the rally engine (fully pixel-space, no court setup), the cutter
/// needs world coordinates for the near/far ready-band, so it takes a
/// court-corner Homography (resurrected in geometry.dart).  Corners come from
/// the one-time calibration screen (Phase 4).
///
/// Two channels run the ball model on a CROP in the Python pipeline and are
/// isolated behind interfaces here, because the Dart OnnxDetector feeds a
/// fixed [1,3,960,960] tensor and whether the exported ONNX accepts a smaller
/// crop input is a device-side unknown:
///   • toss  (near-serve signal) — see [TossSource]
///   • fballs (far-serve ball trace) — see [FarBallSource]
/// The extractor is complete and faithful for everything else; swapping in
/// crop-based implementations of these two does not change it.

/// Version tag written into the meta header — must track TELEMETRY_VERSION in
/// match_telemetry.py (4, ftoss dropped).
const int kTelemetryVersion = 4;

// ---------------------------------------------------------------------------
// Per-frame record + JSONL schema (must match match_telemetry.py exactly)
// ---------------------------------------------------------------------------

class MatchFrameRecord {
  final int f;
  final double t;
  final List<int>? nearBox; // [x1,y1,x2,y2] (analysis px)
  final List<double>? nearWorld; // [wx, wy] (court feet)
  final List<int>? farBox;
  final bool farHeld; // far box carried through a detection gap
  final List<double>? farWorld;
  final List<Detection> balls;
  final List<Detection> toss;
  final List<Detection> fballs;
  final double trophy; // 0.0 — trophy model not ported (stage 2 optional)
  final double stgcn; // 0.0 — ST-GCN not ported (stage 2 ignores it)

  const MatchFrameRecord({
    required this.f,
    required this.t,
    required this.nearBox,
    required this.nearWorld,
    required this.farBox,
    required this.farHeld,
    required this.farWorld,
    required this.balls,
    required this.toss,
    required this.fballs,
    this.trophy = 0.0,
    this.stgcn = 0.0,
  });

  static List<List<double>> _dets(List<Detection> ds) =>
      [for (final d in ds) [_r1(d.x), _r1(d.y), _r3(d.conf)]];

  /// JSONL object identical to the Python record (keys and rounding).
  Map<String, dynamic> toJson() => {
        'f': f,
        't': _rN(t, 4),
        'np': nearBox,
        'npw': nearWorld == null
            ? null
            : [_r2(nearWorld![0]), _r2(nearWorld![1])],
        'fp': farBox,
        'fph': farHeld ? 1 : 0,
        'fpw':
            farWorld == null ? null : [_r2(farWorld![0]), _r2(farWorld![1])],
        'balls': _dets(balls),
        'toss': _dets(toss),
        'fballs': _dets(fballs),
        'trophy': _r3(trophy),
        'stgcn': _r3(stgcn),
      };
}

double _rN(double v, int n) {
  final m = _pow10(n);
  return (v * m).round() / m;
}

double _r1(double v) => _rN(v, 1);
double _r2(double v) => _rN(v, 2);
double _r3(double v) => _rN(v, 3);

double _pow10(int n) {
  var r = 1.0;
  for (var i = 0; i < n; i++) {
    r *= 10;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Crop-dependent channels behind interfaces
// ---------------------------------------------------------------------------

/// Near-serve toss candidates (a ball rising above the near player).  The
/// Python pipeline runs the ball model on a dedicated ROI crop above the near
/// player at imgsz=320 for sensitivity; [WholeFrameTossSource] instead reuses
/// the whole-frame ball detections filtered to the same acceptance zone,
/// which needs no extra model call and no variable-size ONNX input.  A future
/// crop-based source (Python parity, higher toss recall — the binding
/// constraint on near-serve detection) can replace this without touching the
/// extractor.
abstract class TossSource {
  Future<List<Detection>> toss(
    Uint8List rgb,
    List<DetBox> wholeFrameBalls,
    List<int>? nearBox,
    List<List<int>> exclusionZones,
  );
}

/// v1 toss source: filter the whole-frame ball detections into the toss zone
/// above the near player (zone geometry mirrors _detect_toss).  Faithful in
/// its ACCEPTANCE rule; may under-detect vs the Python ROI crop — quantify the
/// gap with the golden-master test before shipping.
class WholeFrameTossSource implements TossSource {
  const WholeFrameTossSource();

  @override
  Future<List<Detection>> toss(
    Uint8List rgb,
    List<DetBox> wholeFrameBalls,
    List<int>? nearBox,
    List<List<int>> exclusionZones,
  ) async {
    if (nearBox == null) return const [];
    final nx1 = nearBox[0].toDouble(), ny1 = nearBox[1].toDouble();
    final nx2 = nearBox[2].toDouble(), ny2 = nearBox[3].toDouble();
    final pw = nx2 - nx1, ph = ny2 - ny1;
    if (pw <= 0 || ph <= 0) return const [];

    // Zone box: bottom bisects the player box; 2x width, 1.5x height.
    final pcx = (nx1 + nx2) / 2.0, pcy = (ny1 + ny2) / 2.0;
    final zx1 = pcx - pw, zx2 = pcx + pw;
    final zy2 = pcy, zy1 = (zy2 - ph * 1.5).clamp(0.0, double.infinity);

    final out = <Detection>[];
    for (final b in wholeFrameBalls) {
      final cx = b.cx, cy = b.cy;
      final inZone = cx >= zx1 && cx <= zx2 && cy >= zy1 && cy <= zy2;
      final inPbox =
          cx >= nx1 - 15 && cx <= nx2 + 15 && cy >= ny1 - 15 && cy <= ny2 + 15;
      if (inZone && !inPbox && !isInExclusionZone(cx, cy, exclusionZones)) {
        out.add(Detection(_r1(cx), _r1(cy), _r3(b.conf)));
      }
    }
    return out;
  }
}

/// Axis-aligned crop rectangle in analysis (960×540) pixel space.
class CropRect {
  final double x1, y1, x2, y2;
  const CropRect(this.x1, this.y1, this.x2, this.y2);
  double get width => x2 - x1;
  double get height => y2 - y1;
}

/// The FIXED far-region crop for fballs (user decision, 2026-07): the bounding
/// box of the far HALF of the court — net line (world y = L/2) to far baseline
/// (world y = L) — projected to pixel space via the calibrated homography,
/// with the top edge extended upward by [CutterConfig.farCropTopExtendFrac] of
/// the box height to reach above the far baseline.  Computed once at
/// calibration from [pixelCornersBlBrTrTl] (the clicked corners, in the
/// calibration order BL, BR, TR, TL); no per-frame far-player-box dependency.
///
/// Note: the far half is heavily foreshortened (a ~20-45px sliver near the
/// frame top on the ground-truth cameras), so the default 0.5 extension only
/// reaches ~10px above the far baseline.  If a stage-1 re-extract shows
/// far-serve recall dropping vs the Python player-anchored crop (16/22 on
/// folder 68), raise farCropTopExtendFrac to reach the toss/contact region.
CropRect farCourtCropRect(List<List<double>> pixelCornersBlBrTrTl,
    {double topExtendFrac = CutterConfig.farCropTopExtendFrac}) {
  const w = CutterConfig.courtWidthFt;
  const l = CutterConfig.courtLengthFt;
  // World corners in the SAME order as the clicked pixel corners.
  final worldCorners = <List<double>>[
    [0.0, 0.0], // BL
    [w, 0.0], // BR
    [w, l], // TR
    [0.0, l], // TL
  ];
  final worldToPixel = Homography.from4(worldCorners, pixelCornersBlBrTrTl);
  // Far half: net line (y = L/2) to far baseline (y = L).
  final farWorld = <List<double>>[
    [0.0, l / 2], [w, l / 2], // net line
    [w, l], [0.0, l], // far baseline
  ];
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (final p in farWorld) {
    final px = worldToPixel.transform(p[0], p[1]);
    minX = math.min(minX, px[0]);
    maxX = math.max(maxX, px[0]);
    minY = math.min(minY, px[1]);
    maxY = math.max(maxY, px[1]);
  }
  minY -= topExtendFrac * (maxY - minY); // extend top upward
  const fw = EngineConfig.analysisWidth, fh = EngineConfig.analysisHeight;
  return CropRect(
    minX.clamp(0.0, fw.toDouble()),
    minY.clamp(0.0, fh.toDouble()),
    maxX.clamp(0.0, fw.toDouble()),
    maxY.clamp(0.0, fh.toDouble()),
  );
}

/// Far-region native-resolution ball candidates (the far-serve trace signal).
/// The far-baseline ball is 1–3 px in the 960×540 analysis frame — below the
/// detection floor — so these come from a native-resolution crop of the fixed
/// far-court rectangle ([farCourtCropRect]).
///
/// On the ONNX-input question (unresolved on-device): the concrete source does
/// NOT feed a crop-sized tensor.  It takes the native-res crop and LETTERBOXES
/// it into the model's existing 960×960 input — which works whether or not the
/// exported ONNX accepts a variable input size, and is exactly what magnifies
/// the tiny far ball.  The one remaining dependency is native-res pixel access
/// for the fixed rectangle (the 960×540 [FrameSource] can't provide it) —
/// a second ffmpeg `crop` output at native res, streamed in parallel.
/// [NoFarBalls] is the stub until that frame-source plumbing lands.
abstract class FarBallSource {
  Future<List<Detection>> farBalls(
    double tSec,
    List<int>? farBox,
    List<double>? farWorld,
    List<List<int>> exclusionZones,
  );
}

/// Stub: no far-region native detections.  Far serves that depend on the
/// native crop will not form a trace until a concrete source is wired up.
class NoFarBalls implements FarBallSource {
  const NoFarBalls();

  @override
  Future<List<Detection>> farBalls(
    double tSec,
    List<int>? farBox,
    List<double>? farWorld,
    List<List<int>> exclusionZones,
  ) async =>
      const [];
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Result of tracking both players on one frame.
class _Players {
  final List<int>? nearBox;
  final List<double>? nearWorld;
  final List<int>? farBox;
  final List<double>? farWorldRaw;
  const _Players(this.nearBox, this.nearWorld, this.farBox, this.farWorldRaw);
}

class MatchTelemetryExtractor {
  final OnnxDetector playerDetector;
  final OnnxDetector ballDetector;
  final Homography homography;
  final List<List<double>> activeZone; // polygon in 960×540 space
  final List<List<int>> exclusionZones;
  final double fps;
  final TossSource tossSource;
  final FarBallSource farBallSource;

  // Far-box hold + world-smoothing state (mirrors the extractor).
  List<int>? _lastFarBox;
  double _lastFarBoxT = -1e9;
  final List<List<double>> _farWorldHistory = []; // (t, wx, wy)

  MatchTelemetryExtractor({
    required this.playerDetector,
    required this.ballDetector,
    required this.homography,
    required this.activeZone,
    required this.exclusionZones,
    required this.fps,
    this.tossSource = const WholeFrameTossSource(),
    this.farBallSource = const NoFarBalls(),
  });

  /// The meta header — first JSONL line, matching match_telemetry.py.
  Map<String, dynamic> meta(int totalFrames, {required bool hasFarBalls}) =>
      metaHeader(fps, totalFrames, hasFarBalls: hasFarBalls);

  /// Meta header as a standalone (no extractor instance needed).
  static Map<String, dynamic> metaHeader(double fps, int totalFrames,
          {required bool hasFarBalls}) =>
      {
        'meta': {
          'version': kTelemetryVersion,
          'fps': fps,
          'total_frames': totalFrames,
          'stride': 1,
          'analysis_size': [
            EngineConfig.analysisWidth,
            EngineConfig.analysisHeight
          ],
          'court_length_ft': CutterConfig.courtLengthFt,
          'court_width_ft': CutterConfig.courtWidthFt,
          'has_trophy': false, // trophy model not ported
          'has_far_serve': false, // ST-GCN not ported
          'has_far_balls': hasFarBalls,
        }
      };

  /// One JSONL line for [record].
  static String encodeRecord(MatchFrameRecord record) =>
      jsonEncode(record.toJson());

  static String encodeMeta(Map<String, dynamic> m) => jsonEncode(m);

  /// Process one 960×540 rgb24 frame into a telemetry record.
  Future<MatchFrameRecord> processFrame(
      Uint8List rgb, double tSec, int frameId) async {
    final tensor = letterboxRgbToTensor(rgb);

    final players = _trackPlayers(tensor);
    var farBox = players.farBox;

    // Far-box hold through short detection gaps.
    var farHeld = false;
    if (farBox != null) {
      _lastFarBox = farBox;
      _lastFarBoxT = tSec;
    } else if (_lastFarBox != null &&
        tSec - _lastFarBoxT <= CutterConfig.farBoxHoldS) {
      farBox = _lastFarBox;
      farHeld = true;
    }

    final farWorld = _smoothedFarWorld(players.farWorldRaw, tSec);

    // One whole-frame ball pass shared by `balls` (active-zone) and the
    // whole-frame toss source (toss-zone).
    final rawBalls =
        ballDetector.detect(tensor, CutterConfig.ballConf, classIndex: CutterConfig.ballClassIndex);
    final balls = <Detection>[];
    for (final b in rawBalls) {
      final cx = b.cx, cy = b.cy;
      if (pointInPolygon(cx, cy, activeZone) &&
          !isInExclusionZone(cx, cy, exclusionZones)) {
        balls.add(Detection(_r1(cx), _r1(cy), _r3(b.conf)));
      }
    }

    final toss =
        await tossSource.toss(rgb, rawBalls, players.nearBox, exclusionZones);
    final fballs =
        await farBallSource.farBalls(tSec, farBox, farWorld, exclusionZones);

    return MatchFrameRecord(
      f: frameId,
      t: tSec,
      nearBox: players.nearBox,
      nearWorld: players.nearWorld,
      farBox: farBox,
      farHeld: farHeld,
      farWorld: farWorld,
      balls: balls,
      toss: toss,
      fballs: fballs,
    );
  }

  /// Near/far player classification with world coordinates (mirrors
  /// _track_players): near = feet closest to the near baseline (world y≈0),
  /// far = feet closest to the far baseline (world y≈court length), each with
  /// feet-x inside the padded sidelines.
  _Players _trackPlayers(Float32List tensor) {
    final boxes = playerDetector.detect(tensor, CutterConfig.playerConf,
        classIndex: CutterConfig.playerClassIndex);
    if (boxes.isEmpty) return const _Players(null, null, null, null);

    const l = CutterConfig.courtLengthFt;
    const w = CutterConfig.courtWidthFt;

    // (box, wx, wy, conf) for each candidate, world at feet (cx, y2).
    final cands = <_Cand>[];
    for (final b in boxes) {
      final cx = (b.x1 + b.x2) / 2.0;
      final world = homography.transform(cx, b.y2);
      cands.add(_Cand(b, world[0], world[1], b.conf));
    }

    // Near: conf floor, closer to near baseline than far, feet-x in band.
    _Cand? near;
    for (final c in cands) {
      if (c.conf < CutterConfig.nearMinConf) continue;
      if (!(c.wy.abs() < (c.wy - l).abs())) continue;
      if (c.wx < -CutterConfig.nearPlayerXPadFt ||
          c.wx > w + CutterConfig.nearPlayerXPadFt) {
        continue;
      }
      if (near == null || c.wy.abs() < near.wy.abs()) near = c;
    }
    final nearBox = near == null ? null : _boxInts(near.b);
    final nearWorld = near == null ? null : [near.wx, near.wy];

    // Far: closer to far baseline, feet-x in band, not the near box.
    _Cand? far;
    for (final c in cands) {
      if (!((c.wy - l).abs() < c.wy.abs())) continue;
      if (c.wx < -CutterConfig.farPlayerXPadFt ||
          c.wx > w + CutterConfig.farPlayerXPadFt) {
        continue;
      }
      if (near != null && identical(c.b, near.b)) continue;
      if (far == null || (c.wy - l).abs() < (far.wy - l).abs()) far = c;
    }
    final farBox = far == null ? null : _boxInts(far.b);
    final farWorldRaw = far == null ? null : [far.wx, far.wy];

    return _Players(nearBox, nearWorld, farBox, farWorldRaw);
  }

  /// Rolling mean of far-player world position over farWorldSmoothS seconds —
  /// the homography amplifies far-court feet-pixel jitter into feet of world
  /// noise.  Returns null when no far position is in the window.
  List<double>? _smoothedFarWorld(List<double>? raw, double now) {
    if (raw != null) {
      _farWorldHistory.add([now, raw[0], raw[1]]);
    }
    while (_farWorldHistory.isNotEmpty &&
        now - _farWorldHistory.first[0] > CutterConfig.farWorldSmoothS) {
      _farWorldHistory.removeAt(0);
    }
    if (_farWorldHistory.isEmpty) return null;
    var sx = 0.0, sy = 0.0;
    for (final e in _farWorldHistory) {
      sx += e[1];
      sy += e[2];
    }
    final n = _farWorldHistory.length;
    return [sx / n, sy / n];
  }

  static List<int> _boxInts(DetBox b) =>
      [b.x1.round(), b.y1.round(), b.x2.round(), b.y2.round()];
}

class _Cand {
  final DetBox b;
  final double wx, wy, conf;
  const _Cand(this.b, this.wx, this.wy, this.conf);
}
