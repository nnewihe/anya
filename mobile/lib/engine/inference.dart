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
    try {
      opts.appendCoreMLProvider(CoreMLFlags.useNone);
    } catch (_) {
      // CoreML unavailable (non-Apple / desktop Linux) — fall back to CPU.
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
