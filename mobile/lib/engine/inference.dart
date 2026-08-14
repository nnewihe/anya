import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

import 'config.dart';
import 'ort_fast_output.dart';

/// A decoded detection box in 960×540 analysis-frame pixel space.
class DetBox {
  final double x1, y1, x2, y2, conf;
  final int cls;
  const DetBox(this.x1, this.y1, this.x2, this.y2, this.conf, this.cls);

  double get cx => (x1 + x2) / 2.0;
  double get cy => (y1 + y2) / 2.0;

  double iou(DetBox o) {
    final ix1 = math.max(x1, o.x1), iy1 = math.max(y1, o.y1);
    final ix2 = math.min(x2, o.x2), iy2 = math.min(y2, o.y2);
    final iw = math.max(0.0, ix2 - ix1), ih = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;
    final ua = (x2 - x1) * (y2 - y1) + (o.x2 - o.x1) * (o.y2 - o.y1) - inter;
    return ua > 0 ? inter / ua : 0.0;
  }
}

/// Letterbox params for the rectangular 960×544 whole-frame model input
/// (960×540 native frame → 960×544 padded, r=1.0, dw=0, dh=2 top/2 bottom).
/// Empirically verified against ultralytics.data.augment.LetterBox — see
/// spikes/fixtures/rect_letterbox_meta.json.
const int _fw = EngineConfig.analysisWidth; // native frame width (960)
const int _fh = EngineConfig.analysisHeight; // native frame height (540)
const int _mw = EngineConfig.modelWidth; // rect model input width (960)
const int _mh = EngineConfig.modelHeight; // rect model input height (544)
const int _padTop = EngineConfig.modelPadTop; // 2

/// Square far-crop model input size (unchanged from the original spike).
const int _sz = EngineConfig.imgsz; // 960

/// Reusable per-isolate scratch buffer for [letterboxRgbToTensor] — avoids
/// re-allocating + re-filling an 11 MB Float32List on every frame (was the
/// single largest allocation churn source in the hot loop; see the padding
/// pre-fill below, which now only needs to run once).
final Float32List _rectTensorBuf = _makeRectBuf();
Float32List _makeRectBuf() {
  final buf = Float32List(3 * _mw * _mh);
  const pad = 114 / 255.0;
  for (var i = 0; i < buf.length; i++) {
    buf[i] = pad;
  }
  return buf;
}

/// 256-entry [0,255] -> [0,1] lookup table, replacing a division per pixel
/// channel (3 divisions/pixel x 518,400 pixels/frame otherwise).
final Float32List _byteToUnit = Float32List.fromList(
    [for (var i = 0; i < 256; i++) i / 255.0]);

/// Convert a 960×540 rgb24 frame (row-major, 3 bytes/pixel) to the rect
/// model's [1,3,544,960] float tensor with gray(114) letterbox padding
/// (2px top/bottom). This exact recipe was validated against Ultralytics in
/// spikes/export_mobile_models.py (worst_iou=1.0, worst_conf_delta=0.0).
///
/// Returns a buffer OWNED BY THIS FUNCTION (reused every call) — callers
/// must finish using it (i.e. hand it to ORT) before calling again. This
/// matches how the hot loop uses it: build tensor -> run inference -> done.
Float32List letterboxRgbToTensor(Uint8List rgb) {
  final out = _rectTensorBuf;
  const plane = _mw * _mh;
  for (var y = 0; y < _fh; y++) {
    final ty = y + _padTop;
    final rowBase = ty * _mw;
    final srcRow = y * _fw * 3;
    for (var x = 0; x < _fw; x++) {
      final p = srcRow + x * 3;
      final base = rowBase + x;
      out[base] = _byteToUnit[rgb[p]];
      out[plane + base] = _byteToUnit[rgb[p + 1]];
      out[2 * plane + base] = _byteToUnit[rgb[p + 2]];
    }
  }
  return out;
}

/// Wraps one ONNX model (players or ball). Two decode strategies:
///  * end-to-end (player model, both variants): fixed [1,300,6] output, no
///    NMS needed (already the model's global top-300 by score).
///  * raw (ball model, both variants): [1,5,N] output (cx,cy,w,h,score per
///    anchor, single class) — NMS-free by design (spikes/FINDINGS.md
///    rejected the baked-NMS export because NonMaxSuppression/TopK force
///    CPU fallback on CoreML/NNAPI). NMS runs here in Dart, ported verbatim
///    from the validated spikes/onnx_dart/lib/main.dart::_decodeRaw/_nms.
/// Output tensors are read via [readFloat32TensorFast] (a raw buffer copy)
/// instead of OrtValue.value's per-element FFI unboxing + List.reshape,
/// which spikes/FINDINGS.md measured at ~100ms/frame for the ball model's
/// output — the dominant cost in the whole pipeline.
enum DetectorKind { end2end, raw }

class OnnxDetector {
  final OrtSession _session;
  final String _inputName;
  final int inputW, inputH;
  final DetectorKind kind;
  final int rawN; // grid length for DetectorKind.raw; unused for end2end
  static const double _nmsIouThresh = 0.45;
  int _refCount = 1;
  String? _sharedCacheKey;

  OnnxDetector._(this._session, this._inputName, this.inputW, this.inputH,
      this.kind, this.rawN);

  /// Per-isolate cache for [fromAssetShared] — Engine and DeadTimeEngine load
  /// the SAME yolo26n.onnx/ball_best.onnx assets; if both are ever
  /// instantiated in one process (isolate), this avoids loading two
  /// independent ONNX sessions (and the ANE/NNAPI compilation cost that goes
  /// with each) for identical model+shape+decode configs. Not currently
  /// exercised — DeadTimeEngine isn't wired to any screen yet — but harmless
  /// and correct when it is: reference-counted so disposing one engine
  /// doesn't invalidate a session the other engine still holds.
  static final Map<String, Future<OnnxDetector>> _sharedCache = {};

  static Future<OnnxDetector> fromAssetShared(
    String assetPath, {
    required int inputW,
    required int inputH,
    required DetectorKind kind,
    int rawN = 0,
  }) {
    final existing = _sharedCache[assetPath];
    if (existing != null) {
      return existing.then((d) {
        d._refCount++;
        return d;
      });
    }
    final future = fromAsset(assetPath,
            inputW: inputW, inputH: inputH, kind: kind, rawN: rawN)
        .then((d) {
      d._sharedCacheKey = assetPath;
      return d;
    });
    _sharedCache[assetPath] = future;
    return future;
  }

  static Future<OnnxDetector> fromAsset(
    String assetPath, {
    required int inputW,
    required int inputH,
    required DetectorKind kind,
    int rawN = 0,
  }) async {
    final bytes = (await rootBundle.load(assetPath)).buffer.asUint8List();
    return fromBytes(bytes,
        inputW: inputW, inputH: inputH, kind: kind, rawN: rawN);
  }

  /// Like [fromAsset] but takes already-loaded model bytes. Use this from a
  /// background isolate: `rootBundle.load` needs a full Flutter binding
  /// (ServicesBinding.instance), which `BackgroundIsolateBinaryMessenger`
  /// does NOT provide — only the binary messenger, not the whole binding —
  /// so asset bytes must be read on the main isolate and handed over (see
  /// engine_isolate.dart). `OrtSession.fromBuffer` itself is pure FFI and has
  /// no such requirement.
  static Future<OnnxDetector> fromBytes(
    Uint8List bytes, {
    required int inputW,
    required int inputH,
    required DetectorKind kind,
    int rawN = 0,
  }) async {
    OrtEnv.instance.init();
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(4);
    // Native-ML execution providers: the Dart code stays cross-platform and
    // ONNX Runtime delegates inference to the OS's native ML stack —
    //   • Apple (iOS/macOS): CoreML EP → Apple Neural Engine / GPU
    //   • Android: NNAPI EP → vendor NPU / GPU / DSP
    // Anything the native EP can't run (or non-Apple/Android desktops) falls
    // back to ORT's CPU kernels automatically. Both exported graphs are now
    // NMS-free (no NonMaxSuppression/TopK-with-dynamic-shape nodes), so the
    // whole graph is CoreML/NNAPI-eligible instead of partially falling back
    // to CPU for the NMS subgraph.
    //
    // iOS specifically: `onlyEnableDeviceWithANE` skips CoreML entirely on
    // devices with no Neural Engine (so inference falls back to ORT's plain
    // CPU EP instead of CoreML-via-GPU on those devices) — iOS disallows GPU
    // work while backgrounded, so a GPU-routed CoreML graph would stall or
    // get the app killed once backgrounded, while ANE/CPU keep running. Note
    // this DOESN'T force "ANE+CPU only, never GPU" on ANE-equipped devices —
    // the onnxruntime Dart package (1.4.1) only wraps ORT's legacy CoreML
    // bitmask flags, not the newer `MLComputeUnits` provider-options API
    // that would let CoreML itself be constrained that precisely; that would
    // need a package fork/patch.
    try {
      if (Platform.isIOS) {
        opts.appendCoreMLProvider(CoreMLFlags.onlyEnableDeviceWithANE);
      } else if (Platform.isMacOS) {
        opts.appendCoreMLProvider(CoreMLFlags.useNone);
      } else if (Platform.isAndroid) {
        opts.appendNnapiProvider(NnapiFlags.useNone);
      }
    } catch (_) {
      // Native EP unavailable — CPU fallback.
    }
    final session = OrtSession.fromBuffer(bytes, opts);
    return OnnxDetector._(
        session, session.inputNames.first, inputW, inputH, kind, rawN);
  }

  Float32List _runRaw(Float32List tensor) {
    final input =
        OrtValueTensor.createTensorWithDataList(tensor, [1, 3, inputH, inputW]);
    final outputs = _session.run(OrtRunOptions(), {_inputName: input});
    final n = kind == DetectorKind.raw
        ? 5 * rawN
        : EngineConfig.playerOutputRows * 6;
    final data = readFloat32TensorFast(outputs.first!, n);
    input.release();
    for (final o in outputs) {
      o?.release();
    }
    return data;
  }

  List<DetBox> _decodeEnd2End(Float32List flat, double conf, bool unletter) {
    final out = <DetBox>[];
    for (var i = 0; i < EngineConfig.playerOutputRows; i++) {
      final base = i * 6;
      final c = flat[base + 4];
      if (c < conf) continue;
      final cls = flat[base + 5].round();
      final x1 = flat[base], y1 = flat[base + 1];
      final x2 = flat[base + 2], y2 = flat[base + 3];
      out.add(unletter
          ? DetBox(x1, y1 - _padTop, x2, y2 - _padTop, c, cls)
          : DetBox(x1, y1, x2, y2, c, cls));
    }
    return out;
  }

  List<DetBox> _decodeRawGrid(Float32List flat, double conf, bool unletter) {
    final n = rawN;
    final cand = <DetBox>[];
    for (var j = 0; j < n; j++) {
      final s = flat[4 * n + j]; // row 4 = class score (single class)
      if (s < conf) continue;
      final cx = flat[j], cy = flat[n + j];
      final w = flat[2 * n + j], h = flat[3 * n + j];
      final x1 = cx - w / 2, y1 = cy - h / 2;
      final x2 = cx + w / 2, y2 = cy + h / 2;
      cand.add(unletter
          ? DetBox(x1, y1 - _padTop, x2, y2 - _padTop, s, 0)
          : DetBox(x1, y1, x2, y2, s, 0));
    }
    cand.sort((a, b) => b.conf.compareTo(a.conf));
    return _nms(cand, _nmsIouThresh);
  }

  List<DetBox> _nms(List<DetBox> boxes, double iouThr) {
    final keep = <DetBox>[];
    final rest = List<DetBox>.from(boxes);
    while (rest.isNotEmpty) {
      final b = rest.removeAt(0);
      keep.add(b);
      rest.removeWhere((o) => b.iou(o) >= iouThr);
    }
    return keep;
  }

  /// Run detection on a preprocessed tensor; keep boxes with conf >= [conf]
  /// and (if given) class == [classIndex]. Un-letterboxes back to 960×540
  /// analysis-frame space (only meaningful for tensors built by
  /// [letterboxRgbToTensor] — the rect whole-frame path).
  List<DetBox> detect(Float32List tensor, double conf, {int? classIndex}) {
    final flat = _runRaw(tensor);
    final boxes = kind == DetectorKind.raw
        ? _decodeRawGrid(flat, conf, true)
        : _decodeEnd2End(flat, conf, true);
    return classIndex == null
        ? boxes
        : [for (final b in boxes) if (b.cls == classIndex) b];
  }

  /// Like [detect] but returns boxes in RAW letterbox coordinates (no
  /// un-letterboxing). Used for the native-res far-ball crop, which is
  /// letterboxed with a different transform than the fixed whole-frame one
  /// — the caller un-letterboxes with its own params.
  List<DetBox> detectRaw(Float32List tensor, double conf, {int? classIndex}) {
    final flat = _runRaw(tensor);
    final boxes = kind == DetectorKind.raw
        ? _decodeRawGrid(flat, conf, false)
        : _decodeEnd2End(flat, conf, false);
    return classIndex == null
        ? boxes
        : [for (final b in boxes) if (b.cls == classIndex) b];
  }

  /// Releases the underlying ONNX session, unless it's shared (loaded via
  /// [fromAssetShared]) and another owner still holds it.
  void dispose() {
    _refCount--;
    if (_refCount > 0) return;
    if (_sharedCacheKey != null) {
      _sharedCache.remove(_sharedCacheKey);
    }
    _session.release();
  }
}

/// The uniform-scale + centered-pad transform mapping a native crop's pixels
/// into the model's 960×960 input, so detections can be mapped back out.
class LetterboxTransform {
  final double scale; // native-crop px -> letterbox px
  final double padX, padY;
  const LetterboxTransform(this.scale, this.padX, this.padY);
}

/// Letterbox an arbitrary [w]×[h] rgb24 crop into the SQUARE far-crop model's
/// [1,3,960,960] tensor (uniform scale to fit, gray-114 pad, bilinear),
/// returning the transform for mapping detections back. Unlike
/// [letterboxRgbToTensor] (fixed rect 960×544), this handles any crop size —
/// the native-res far crop — and always targets the square [_sz] used by
/// ball_best_far_crop.onnx, independent of the whole-frame rect optimization.
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
