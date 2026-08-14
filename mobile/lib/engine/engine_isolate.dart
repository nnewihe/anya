import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' show RootIsolateToken;

import 'package:flutter/services.dart'
    show BackgroundIsolateBinaryMessenger, rootBundle;

import 'engine.dart';
import 'frame_source.dart' show VideoInfo;
import 'platform.dart' show isMobilePlatform;

/// Runs [Engine.analyze] on a dedicated background isolate so ONNX inference
/// never blocks the UI/main isolate — previously the entire hot loop
/// (letterbox, `session.run`, tracking) ran synchronously on whichever
/// isolate called `Engine.shared().analyze(...)` (the UI isolate, from
/// match_setup_screen.dart), causing jank while foregrounded and tying the
/// work to the UI isolate's OS scheduling priority while backgrounded.
///
/// **Desktop only** (`!isMobilePlatform`). On a real iOS simulator this was
/// tested and CRASHES: `ffmpeg_kit_flutter_new`'s init sequence
/// (`FFmpegKitInitializer._initialize()` — an `EventChannel` broadcast
/// subscription immediately followed by an awaited `MethodChannel` call,
/// `getLogLevel`) fatals with
/// `[FATAL] Check failed: did_send.` in
/// `platform_message_response_dart_port.cc` when run for the first time from
/// a background isolate (even with `BackgroundIsolateBinaryMessenger`
/// correctly initialized) — reproduced with `flutter run` on an iPhone 16
/// Pro simulator, iOS 18 / Flutter 3.44.4. The IDENTICAL `Engine.analyze()`
/// call on the ROOT isolate (the pre-existing behavior, see
/// analyze_ios_main.dart) does not crash. Desktop's `FfmpegFrameSource` uses
/// `dart:io` Process directly — no platform channel involved in decode —
/// and was verified safe (two full end-to-end runs on macOS; see
/// spikes/verify_ball_recall.py-adjacent benchmark checkpoints in the perf
/// report). So: offload on desktop, run in-place on mobile until the
/// ffmpeg_kit/background-isolate interaction is root-caused (or the plugin
/// is swapped) — see the perf report's "Known limitation" section for the
/// follow-up architecture (stream decoded frames INTO a worker isolate that
/// owns only the ONNX sessions, keeping all ffmpeg_kit calls on the root
/// isolate) that would fix this properly.
///
/// The worker isolate loads its OWN independent [Engine] (model loading is
/// isolate-local; `Engine.shared()`'s cached Future lives only in the calling
/// isolate's memory) and needs [BackgroundIsolateBinaryMessenger] initialized
/// before touching any platform channel — asset loading (`rootBundle`, used
/// by `OnnxDetector.fromAsset`) goes through one.
///
/// Messages crossing the isolate boundary are plain Maps/Lists/primitives
/// (not the custom EngineResult/RallySegment/VideoInfo classes) to avoid any
/// ambiguity about which Dart object graphs are isolate-sendable.
Future<EngineResult> analyzeInBackground({
  required String videoPath,
  String? reelPath,
  bool writeReel = true,
  void Function(double fraction, String message)? onProgress,
}) async {
  if (isMobilePlatform) {
    final engine = await Engine.shared();
    return engine.analyze(
      videoPath: videoPath,
      reelPath: reelPath,
      writeReel: writeReel,
      onProgress: onProgress,
    );
  }

  final token = RootIsolateToken.instance;
  if (token == null) {
    throw StateError('analyzeInBackground must be called from the root isolate');
  }
  // Model bytes must be read HERE, on the calling (root) isolate:
  // rootBundle.load needs a full Flutter binding (ServicesBinding.instance),
  // which a spawned isolate doesn't have even after
  // BackgroundIsolateBinaryMessenger.ensureInitialized (that sets up only the
  // binary messenger, not the whole binding) — see OnnxDetector.fromBytes.
  final playerBytes =
      (await rootBundle.load('assets/models/yolo26n.onnx')).buffer.asUint8List();
  final ballBytes =
      (await rootBundle.load('assets/models/ball_best.onnx')).buffer.asUint8List();

  final receivePort = ReceivePort();
  final errorPort = ReceivePort();
  final exitPort = ReceivePort();

  final isolate = await Isolate.spawn(
    _isolateEntry,
    _AnalyzeRequest(token, videoPath, reelPath, writeReel, playerBytes,
        ballBytes, receivePort.sendPort),
    onError: errorPort.sendPort,
    onExit: exitPort.sendPort,
    errorsAreFatal: true,
  );

  final completer = Completer<EngineResult>();
  late StreamSubscription msgSub;
  late StreamSubscription errSub;
  late StreamSubscription exitSub;
  void cleanup() {
    msgSub.cancel();
    errSub.cancel();
    exitSub.cancel();
    receivePort.close();
    errorPort.close();
    exitPort.close();
    isolate.kill(priority: Isolate.immediate);
  }

  msgSub = receivePort.listen((msg) {
    final map = msg as Map;
    switch (map['type']) {
      case 'progress':
        onProgress?.call(map['fraction'] as double, map['message'] as String);
      case 'result':
        if (!completer.isCompleted) {
          completer.complete(_resultFromMap(map));
        }
        cleanup();
      case 'error':
        if (!completer.isCompleted) {
          completer.completeError(StateError(map['error'] as String));
        }
        cleanup();
    }
  });
  errSub = errorPort.listen((err) {
    if (!completer.isCompleted) {
      completer.completeError(StateError('worker isolate error: $err'));
    }
    cleanup();
  });
  exitSub = exitPort.listen((_) {
    if (!completer.isCompleted) {
      completer.completeError(StateError('worker isolate exited without a result'));
    }
    cleanup();
  });

  return completer.future;
}

class _AnalyzeRequest {
  final RootIsolateToken token;
  final String videoPath;
  final String? reelPath;
  final bool writeReel;
  final Uint8List playerBytes;
  final Uint8List ballBytes;
  final SendPort sendPort;
  const _AnalyzeRequest(this.token, this.videoPath, this.reelPath,
      this.writeReel, this.playerBytes, this.ballBytes, this.sendPort);
}

void _isolateEntry(_AnalyzeRequest req) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(req.token);
  Engine? engine;
  try {
    engine = await Engine.loadFromBytes(
        playerBytes: req.playerBytes, ballBytes: req.ballBytes);
    final result = await engine.analyze(
      videoPath: req.videoPath,
      reelPath: req.reelPath,
      writeReel: req.writeReel,
      onProgress: (fraction, message) => req.sendPort
          .send({'type': 'progress', 'fraction': fraction, 'message': message}),
    );
    req.sendPort.send(_resultToMap(result));
  } catch (e, st) {
    req.sendPort.send({'type': 'error', 'error': '$e\n$st'});
  } finally {
    engine?.dispose();
  }
}

Map<String, dynamic> _resultToMap(EngineResult r) => {
      'type': 'result',
      'segments': [
        for (final s in r.segments) [s.start, s.end, s.origin]
      ],
      'reelPath': r.reelPath,
      'fps': r.videoInfo.fps,
      'totalFrames': r.videoInfo.totalFrames,
      'durationSec': r.videoInfo.durationSec,
    };

EngineResult _resultFromMap(Map map) {
  final segments = [
    for (final s in map['segments'] as List)
      RallySegment((s[0] as num).toDouble(), (s[1] as num).toDouble(), s[2] as String)
  ];
  final info = VideoInfo(
    (map['fps'] as num).toDouble(),
    map['totalFrames'] as int,
    (map['durationSec'] as num).toDouble(),
  );
  return EngineResult(segments, map['reelPath'] as String?, info);
}
