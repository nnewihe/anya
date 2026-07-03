import 'dart:typed_data';

import 'ball_tracker.dart' show Detection;
import 'config.dart';
import 'dbscan.dart';
import 'geometry.dart';
import 'inference.dart';

/// Per-frame telemetry: near/far player boxes and whole-court ball candidates.
/// Mirrors the ACTIVE-state subset of AnyaTelemetryProvider.process_frame —
/// but with NO court homography. Near/far classification is pure pixel-space:
/// the camera sits behind the near baseline, so the near player's feet are
/// lowest in frame and the far player's feet are highest. The active zone
/// (auto-estimated from player feet during the pre-scan) keeps neighbouring
/// courts' players and spectators out of the candidate pool.
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

class TelemetryProvider {
  final OnnxDetector playerDetector;
  final OnnxDetector ballDetector;
  final List<List<double>> activeZone; // polygon in 960×540 space
  final List<List<int>> exclusionZones;
  final double fps;

  static const int _playerStride = 4; // ACTIVE_PLAYER_STRIDE

  /// Minimum vertical feet separation (fraction of frame height) before the
  /// highest-feet candidate is accepted as the FAR player — guards against a
  /// second near-side detection (doubles partner, duplicate box) being
  /// misread as the far player.
  static const double _minFarSeparation = 0.12;

  int _frameCounter = 0;
  ({List<double>? near, List<double>? far})? _cached;
  List<double>? _lastKnownFar;

  TelemetryProvider({
    required this.playerDetector,
    required this.ballDetector,
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
      final tracked = _trackPlayers(tensor);
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

  /// Pixel-space near/far classification (no homography):
  ///   candidates = detections whose FEET land inside the active zone;
  ///   near = feet lowest in frame; far = feet highest, if clearly separated.
  ({List<double>? near, List<double>? far}) _trackPlayers(Float32List tensor) {
    final boxes = playerDetector.detect(tensor, EngineConfig.playerConf,
        classIndex: EngineConfig.playerClassIndex);
    final cands = [
      for (final b in boxes)
        if (pointInPolygon(b.cx, b.y2, activeZone)) b
    ];
    if (cands.isEmpty) return (near: null, far: null);

    cands.sort((a, b) => a.y2.compareTo(b.y2)); // ascending feet-y
    final near = cands.last;
    List<double>? farBox;
    if (cands.length > 1) {
      final far = cands.first;
      if (near.y2 - far.y2 >
          _minFarSeparation * EngineConfig.analysisHeight) {
        farBox = [far.x1, far.y1, far.x2, far.y2];
      }
    }
    return (near: [near.x1, near.y1, near.x2, near.y2], far: farBox);
  }
}
