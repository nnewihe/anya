import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart' show calloc;
import 'package:onnxruntime/onnxruntime.dart';
// Deliberate implementation import: the public OrtValue.value getter
// materializes output tensors by reading them element-by-element through FFI
// into a boxed List<num>, then rebuilding nested Lists via List.reshape
// (onnxruntime-1.4.1/lib/src/{ort_value,util/list_shape_extension}.dart).
// spikes/FINDINGS.md measured that double marshalling at ~100ms/frame for the
// ball model's raw [1,5,N] output — the dominant cost in the whole pipeline,
// well above inference itself. OrtValue.ptr and OrtEnv.instance.ortApiPtr are
// public; this reads the same underlying buffer as a single Float32List view
// with no per-element boxing or reshape pass.
// ignore: implementation_imports
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;

/// Returns the raw float32 output buffer of a fixed-shape tensor as a single
/// flat [Float32List] (length = product of the tensor's dims), skipping the
/// nested-List marshalling that `OrtValue.value` performs. Only valid for
/// tensors whose element type is float32 — true for both the player and ball
/// detector outputs exported by spikes/export_mobile_models.py.
Float32List readFloat32TensorFast(OrtValue value, int length) {
  final dataPtrPtr = calloc<ffi.Pointer<ffi.Float>>();
  final api = OrtEnv.instance.ortApiPtr;
  final statusPtr = api.ref.GetTensorMutableData.asFunction<
          bg.OrtStatusPtr Function(ffi.Pointer<bg.OrtValue>,
              ffi.Pointer<ffi.Pointer<ffi.Void>>)>()(
      value.ptr, dataPtrPtr.cast());
  OrtStatus.checkOrtStatus(statusPtr);
  final dataPtr = dataPtrPtr.value;
  // Copy out of native memory before the OrtValue is released; the source
  // buffer is owned by ONNX Runtime and freed on `outputs[i]?.release()`.
  final out = Float32List.fromList(dataPtr.asTypedList(length));
  calloc.free(dataPtrPtr);
  return out;
}
