import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

import 'config.dart';

/// A decoded detection box in 960×540 analysis-frame pixel space.
class DetBox {
  final double x1, y1, x2, y2, conf;
  final int cls;
  const DetBox(this.x1, this.y1, this.x2, this.y2, this.conf, this.cls);

  double get cx => (x1 + x2) / 2.0;
  double get cy => (y1 + y2) / 2.0;
}

/// Letterbox params for a 960×540 frame → 960×960 square (r=1, pad top/bottom).
const double _lbR = 1.0;
const double _lbDw = 0.0;
const double _lbDh = 210.0;
const int _sz = EngineConfig.imgsz;
const int _fw = EngineConfig.analysisWidth;
const int _fh = EngineConfig.analysisHeight;

/// Convert a 960×540 rgb24 frame (row-major, 3 bytes/pixel) to the model's
/// [1,3,960,960] float tensor with gray(114) letterbox padding. This exact
/// recipe was validated against Ultralytics in the spike.
Float32List letterboxRgbToTensor(Uint8List rgb) {
  final out = Float32List(3 * _sz * _sz);
  const pad = 114 / 255.0;
  for (var i = 0; i < out.length; i++) {
    out[i] = pad;
  }
  const dhi = 210;
  const plane = _sz * _sz;
  for (var y = 0; y < _fh; y++) {
    final ty = y + dhi;
    for (var x = 0; x < _fw; x++) {
      final p = (y * _fw + x) * 3;
      final base = ty * _sz + x;
      out[base] = rgb[p] / 255.0;
      out[plane + base] = rgb[p + 1] / 255.0;
      out[2 * plane + base] = rgb[p + 2] / 255.0;
    }
  }
  return out;
}

/// Wraps one ONNX model (players or ball). Output is the compact end-to-end
/// [1,300,6] tensor (x1,y1,x2,y2,conf,cls); we threshold per call-site in Dart
/// and un-letterbox back to 960×540 space.
class OnnxDetector {
  final OrtSession _session;
  final String _inputName;

  OnnxDetector._(this._session, this._inputName);

  static Future<OnnxDetector> fromAsset(String assetPath) async {
    OrtEnv.instance.init();
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(2);
    // Native-ML execution providers: the Dart code stays cross-platform and
    // ONNX Runtime delegates inference to the OS's native ML stack —
    //   • Apple (iOS/macOS): CoreML EP → Apple Neural Engine / GPU
    //   • Android: NNAPI EP → vendor NPU / GPU / DSP
    // Anything the native EP can't run (or non-Apple/Android desktops) falls
    // back to ORT's CPU kernels automatically.
    try {
      if (Platform.isIOS || Platform.isMacOS) {
        opts.appendCoreMLProvider(CoreMLFlags.useNone);
      } else if (Platform.isAndroid) {
        opts.appendNnapiProvider(NnapiFlags.useNone);
      }
    } catch (_) {
      // Native EP unavailable — CPU fallback.
    }
    final bytes = (await rootBundle.load(assetPath)).buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, opts);
    return OnnxDetector._(session, session.inputNames.first);
  }

  /// Run detection on a preprocessed tensor; keep boxes with conf >= [conf]
  /// and (if given) class == [classIndex].
  List<DetBox> detect(Float32List tensor, double conf, {int? classIndex}) {
    final input =
        OrtValueTensor.createTensorWithDataList(tensor, [1, 3, _sz, _sz]);
    final outputs = _session.run(OrtRunOptions(), {_inputName: input});
    final boxes = <DetBox>[];
    final grid = _rows(outputs.first?.value);
    for (final row in grid) {
      if (row.length < 6) continue;
      final c = row[4];
      if (c < conf) continue;
      final cls = row[5].round();
      if (classIndex != null && cls != classIndex) continue;
      boxes.add(DetBox(
        (row[0] - _lbDw) / _lbR,
        (row[1] - _lbDh) / _lbR,
        (row[2] - _lbDw) / _lbR,
        (row[3] - _lbDh) / _lbR,
        c,
        cls,
      ));
    }
    input.release();
    for (final o in outputs) {
      o?.release();
    }
    return boxes;
  }

  /// Like [detect] but returns boxes in RAW 960×960 letterbox coordinates
  /// (no un-letterboxing).  Used for the native-res far-ball crop, which is
  /// letterboxed with a different transform than the fixed 960×540 one that
  /// [detect] bakes in — the caller un-letterboxes with its own params.
  List<DetBox> detectRaw(Float32List tensor, double conf, {int? classIndex}) {
    final input =
        OrtValueTensor.createTensorWithDataList(tensor, [1, 3, _sz, _sz]);
    final outputs = _session.run(OrtRunOptions(), {_inputName: input});
    final boxes = <DetBox>[];
    for (final row in _rows(outputs.first?.value)) {
      if (row.length < 6) continue;
      final c = row[4];
      if (c < conf) continue;
      final cls = row[5].round();
      if (classIndex != null && cls != classIndex) continue;
      boxes.add(DetBox(row[0], row[1], row[2], row[3], c, cls));
    }
    input.release();
    for (final o in outputs) {
      o?.release();
    }
    return boxes;
  }

  List<List<double>> _rows(dynamic value) {
    // value is [1][N][6]
    final batch = (value as List).first as List;
    return [
      for (final row in batch)
        [for (final v in row as List) (v as num).toDouble()]
    ];
  }

  void dispose() => _session.release();
}

/// The uniform-scale + centered-pad transform mapping a native crop's pixels
/// into the model's 960×960 input, so detections can be mapped back out.
class LetterboxTransform {
  final double scale; // native-crop px -> letterbox px
  final double padX, padY;
  const LetterboxTransform(this.scale, this.padX, this.padY);
}

/// Letterbox an arbitrary [w]×[h] rgb24 crop into the model's [1,3,960,960]
/// tensor (uniform scale to fit, gray-114 pad, bilinear), returning the
/// transform for mapping detections back.  Unlike [letterboxRgbToTensor]
/// (fixed 960×540), this handles any crop size — the native-res far crop.
(Float32List, LetterboxTransform) letterboxCropToTensor(
    Uint8List rgb, int w, int h) {
  final out = Float32List(3 * _sz * _sz);
  const pad = 114 / 255.0;
  for (var i = 0; i < out.length; i++) {
    out[i] = pad;
  }
  final s = math.min(_sz / w, _sz / h);
  final rw = (w * s).round(), rh = (h * s).round();
  final padX = (_sz - rw) / 2.0, padY = (_sz - rh) / 2.0;
  final ox0 = padX.floor(), oy0 = padY.floor();
  const plane = _sz * _sz;
  for (var oy = 0; oy < rh; oy++) {
    final ty = oy + oy0;
    if (ty < 0 || ty >= _sz) continue;
    final syf = (oy + 0.5) / s - 0.5;
    var sy = syf.floor();
    final fy = syf - sy;
    if (sy < 0) sy = 0;
    final sy1 = math.min(sy + 1, h - 1);
    for (var ox = 0; ox < rw; ox++) {
      final tx = ox + ox0;
      if (tx < 0 || tx >= _sz) continue;
      final sxf = (ox + 0.5) / s - 0.5;
      var sx = sxf.floor();
      final fx = sxf - sx;
      if (sx < 0) sx = 0;
      final sx1 = math.min(sx + 1, w - 1);
      final base = ty * _sz + tx;
      for (var ch = 0; ch < 3; ch++) {
        final p00 = rgb[(sy * w + sx) * 3 + ch].toDouble();
        final p01 = rgb[(sy * w + sx1) * 3 + ch].toDouble();
        final p10 = rgb[(sy1 * w + sx) * 3 + ch].toDouble();
        final p11 = rgb[(sy1 * w + sx1) * 3 + ch].toDouble();
        final top = p00 * (1 - fx) + p01 * fx;
        final bot = p10 * (1 - fx) + p11 * fx;
        out[ch * plane + base] = (top * (1 - fy) + bot * fy) / 255.0;
      }
    }
  }
  return (out, LetterboxTransform(s, padX, padY));
}
