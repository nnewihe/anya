import 'dart:typed_data';

import 'ball_tracker.dart' show Detection;
import 'config.dart';
import 'dbscan.dart';
import 'geometry.dart';
import 'inference.dart';

/// Per-frame telemetry: near/far player boxes and whole-court ball candidates.
/// Mirrors the ACTIVE-state subset of AnyaTelemetryProvider.process_frame.
class TelemetryFrame {
  final int frameId;
  final double timestamp;
  final List<double>? nearPlayerBox; // [x1,y1,x2,y2]
  final List<double>? farPlayerBox;
  final List<Detection> activeBallCandidates; // (cx, cy, conf)

  const TelemetryFrame({
    required this.frameId,
    required this.timestamp,
    required this.nearPlayerBox,
    required this.farPlayerBox,
    required this.activeBallCandidates,
  });
}

/// Runs the player + ball detectors each frame and produces telemetry. The
/// engine forces ACTIVE, so only that path is implemented.
class TelemetryProvider {
  final OnnxDetector playerDetector;
  final OnnxDetector ballDetector;
  final Homography homography;
  final List<List<double>> activeZone; // polygon in 960×540 space
  final List<List<int>> exclusionZones;
  final double fps;

  static const int _playerStride = 4; // ACTIVE_PLAYER_STRIDE

  int _frameCounter = 0;
  ({List<double>? near, List<double>? far})? _cached;
  List<double>? _lastKnownFar;

  TelemetryProvider({
    required this.playerDetector,
    required this.ballDetector,
    required this.homography,
    required this.activeZone,
    required this.exclusionZones,
    required this.fps,
  });

  TelemetryFrame processFrame(Float32List tensor) {
    _frameCounter += 1;
    final timestamp = _frameCounter / fps;

    // 1. Player near/far tracking (stride 4, cached in between).
    List<double>? nearBox, farBox;
    if (_frameCounter % _playerStride != 0 && _cached?.near != null) {
      nearBox = _cached!.near;
      farBox = _cached!.far;
    } else {
      final tracked = _trackNearPlayer(tensor);
      nearBox = tracked.near;
      farBox = tracked.far;
      // ACTIVE: persist last known far box across misses.
      if (farBox != null) {
        _lastKnownFar = farBox;
      } else {
        farBox = _lastKnownFar;
      }
      _cached = (near: nearBox, far: farBox);
    }

    // 2. Whole-court ball detection, filtered to active zone / outside exclusions.
    final ballBoxes = ballDetector.detect(tensor, EngineConfig.activeBallConf,
        classIndex: EngineConfig.ballClassIndex);
    final candidates = <Detection>[];
    for (final b in ballBoxes) {
      final cx = b.cx, cy = b.cy;
      if (pointInPolygon(cx, cy, activeZone) &&
          !isInExclusionZone(cx, cy, exclusionZones)) {
        candidates.add(Detection(cx, cy, b.conf));
      }
    }

    return TelemetryFrame(
      frameId: _frameCounter,
      timestamp: timestamp,
      nearPlayerBox: nearBox,
      farPlayerBox: farBox,
      activeBallCandidates: candidates,
    );
  }

  ({List<double>? near, List<double>? far}) _trackNearPlayer(Float32List tensor) {
    final boxes = playerDetector.detect(tensor, EngineConfig.playerConf,
        classIndex: EngineConfig.playerClassIndex);
    if (boxes.isEmpty) return (near: null, far: null);

    // (box, worldX, worldY) using feet = (cx, y2).
    final cands = <({List<double> box, double wx, double wy})>[];
    for (final b in boxes) {
      final cx = (b.x1 + b.x2) / 2.0;
      final world = homography.transform(cx, b.y2);
      cands.add((box: [b.x1, b.y1, b.x2, b.y2], wx: world[0], wy: world[1]));
    }

    const pad = EngineConfig.nearPlayerXPadFt;
    const len = EngineConfig.courtLengthFt;
    const width = EngineConfig.courtWidthFt;
    final nearCands = cands
        .where((c) =>
            c.wy.abs() < (c.wy - len).abs() && // closer to near baseline
            -pad <= c.wx &&
            c.wx <= width + pad)
        .toList();
    if (nearCands.isEmpty) return (near: null, far: null);

    nearCands.sort((a, b) => a.wy.abs().compareTo(b.wy.abs()));
    final near = nearCands.first;

    // Far player: closest to far baseline among everyone who isn't near.
    final rest = cands.where((c) => !_sameBox(c.box, near.box)).toList();
    List<double>? farBox;
    if (rest.isNotEmpty) {
      rest.sort((a, b) => (a.wy - len).abs().compareTo((b.wy - len).abs()));
      farBox = rest.first.box;
    }
    return (near: near.box, far: farBox);
  }

  bool _sameBox(List<double> a, List<double> b) =>
      a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}
