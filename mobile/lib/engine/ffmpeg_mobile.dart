import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:ffmpeg_kit_flutter_new/stream_information.dart';
import 'package:path_provider/path_provider.dart';

import 'config.dart';
import 'frame_source.dart';

const int _w = EngineConfig.analysisWidth;
const int _h = EngineConfig.analysisHeight;
const int _frameBytes = _w * _h * 3;

/// Video encoder for the reel. `ffmpeg_kit_flutter_new` ships a GPL build that
/// includes libx264; if you swap to an LGPL variant, change this to a hardware
/// encoder (e.g. `h264_videotoolbox` on Apple, `h264_mediacodec` on Android).
const String _reelVideoCodec = 'libx264';

Future<VideoInfo> probeMobile(String path) async {
  final session = await FFprobeKit.getMediaInformation(path);
  final mi = session.getMediaInformation();
  var fps = 30.0, dur = 0.0;
  var total = 0;
  if (mi != null) {
    // ffmpeg_kit_flutter_new's typed getters (getStringProperty/
    // getNumberProperty) do an unchecked cast of the raw ffprobe JSON value —
    // ffprobe encodes some numeric fields (e.g. nb_frames) as JSON strings
    // depending on format/version, which throws
    // "type 'String' is not a subtype of type 'num'" from getNumberProperty.
    // Stay on the string getter (frame-rate fields are reliably "a/b"
    // strings) and avoid the numeric getter for nb_frames entirely — total
    // frame count is derived from duration*fps instead, which is exact
    // enough for progress reporting.
    try {
      dur = double.tryParse(mi.getDuration() ?? '') ?? 0.0;
    } catch (_) {}
    try {
      for (final s in mi.getStreams()) {
        if (s.getType() != 'video') continue;
        final rate = s.getStringProperty(StreamInformation.keyRealFrameRate) ??
            s.getStringProperty(StreamInformation.keyAverageFrameRate);
        if (rate != null && rate.contains('/')) {
          final p = rate.split('/');
          final a = double.tryParse(p[0]) ?? 0;
          final b = double.tryParse(p[1]) ?? 1;
          if (a > 0 && b != 0) fps = a / b;
        }
        break;
      }
    } catch (_) {}
  }
  if (fps <= 0 || fps > 300) fps = 30.0;
  if (total <= 0 && dur > 0) total = (dur * fps).round();
  if (dur <= 0 && total > 0) dur = total / fps;
  return VideoInfo(fps, total, dur);
}

/// Mobile [FrameSource] backed by ffmpeg_kit. Decodes the video in bounded
/// windows to a temporary rawvideo file (kept small — [maxFramesPerWindow]
/// frames), streams each frame out, then deletes the window and advances.
/// Decode is not the bottleneck (inference is), so windowing costs little.
class FfmpegKitFrameSource implements FrameSource {
  final String videoPath;
  @override
  final VideoInfo info;
  final int maxFramesPerWindow;
  bool _cancelled = false;

  FfmpegKitFrameSource._(this.videoPath, this.info, this.maxFramesPerWindow);

  static Future<FfmpegKitFrameSource> open(String path,
      {int maxFramesPerWindow = 240}) async {
    return FfmpegKitFrameSource._(path, await probeMobile(path), maxFramesPerWindow);
  }

  @override
  Stream<Uint8List> frames() async* {
    final tmp = await getTemporaryDirectory();
    final winSec = (maxFramesPerWindow / info.fps).clamp(0.5, 30.0);
    final duration = (info.durationSec.isFinite && info.durationSec > 0)
        ? info.durationSec
        : double.infinity;
    var t = 0.0;
    var win = 0;
    while (t < duration && !_cancelled) {
      final raw = '${tmp.path}/anya_win_$win.rgb';
      final session = await FFmpegKit.executeWithArguments([
        '-y',
        '-ss', t.toStringAsFixed(3),
        '-t', winSec.toStringAsFixed(3),
        '-i', videoPath,
        '-vf', 'scale=$_w:$_h',
        '-sws_flags', 'bilinear',
        '-pix_fmt', 'rgb24',
        '-f', 'rawvideo',
        raw,
      ]);
      final f = File(raw);
      if (!ReturnCode.isSuccess(await session.getReturnCode()) || !f.existsSync()) {
        break;
      }
      final len = await f.length();
      if (len < _frameBytes) {
        await _tryDelete(f);
        break;
      }
      final raf = await f.open();
      var off = 0;
      var produced = 0;
      while (off + _frameBytes <= len && !_cancelled) {
        final bytes = await raf.read(_frameBytes);
        if (bytes.length < _frameBytes) break;
        yield bytes;
        off += _frameBytes;
        produced++;
      }
      await raf.close();
      await _tryDelete(f);
      if (produced == 0) break;
      win++;
      t += winSec;
    }
  }

  @override
  Future<void> dispose() async {
    _cancelled = true;
  }
}

/// Cut each merged segment with ffmpeg_kit and concat into [outputPath].
/// [valid] is the already merged/pre-rolled segment list.
Future<bool> reelCutMobile({
  required String videoPath,
  required List<(double start, double end)> valid,
  required String outputPath,
}) async {
  final tmp = await getTemporaryDirectory();
  final dir = await Directory(
          '${tmp.path}/anya_reel_${DateTime.now().microsecondsSinceEpoch}')
      .create();
  try {
    final segFiles = <String>[];
    for (var i = 0; i < valid.length; i++) {
      final (start, end) = valid[i];
      final seg = '${dir.path}/seg_${i.toString().padLeft(4, '0')}.mp4';
      final s = await FFmpegKit.executeWithArguments([
        '-y',
        '-ss', start.toStringAsFixed(3),
        '-to', end.toStringAsFixed(3),
        '-i', videoPath,
        '-c:v', _reelVideoCodec, '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-vsync', 'cfr',
        seg,
      ]);
      if (ReturnCode.isSuccess(await s.getReturnCode())) segFiles.add(seg);
    }
    if (segFiles.isEmpty) return false;

    final list = File('${dir.path}/concat.txt');
    await list.writeAsString(segFiles.map((f) => "file '$f'").join('\n'));
    final s = await FFmpegKit.executeWithArguments([
      '-y',
      '-f', 'concat', '-safe', '0',
      '-i', list.path,
      '-c', 'copy',
      outputPath,
    ]);
    return ReturnCode.isSuccess(await s.getReturnCode());
  } finally {
    await dir.delete(recursive: true).catchError((_) => dir);
  }
}

/// Grab ONE 960×540 rgb24 frame at [tSec] via ffmpeg_kit (random access, for
/// the exclusion-zone scan). Returns null on seek/decode failure.
Future<Uint8List?> grabAnalysisFrameMobile(String videoPath, double tSec) async {
  final tmp = await getTemporaryDirectory();
  final raw =
      '${tmp.path}/anya_grab_${DateTime.now().microsecondsSinceEpoch}.rgb';
  final s = await FFmpegKit.executeWithArguments([
    '-y',
    '-ss', tSec.toStringAsFixed(3),
    '-i', videoPath,
    '-frames:v', '1',
    '-vf', 'scale=$_w:$_h',
    '-sws_flags', 'bilinear',
    '-pix_fmt', 'rgb24',
    '-f', 'rawvideo',
    raw,
  ]);
  final f = File(raw);
  if (!ReturnCode.isSuccess(await s.getReturnCode()) ||
      !f.existsSync() ||
      await f.length() < _frameBytes) {
    await _tryDelete(f);
    return null;
  }
  final bytes = await f.readAsBytes();
  await _tryDelete(f);
  return Uint8List.fromList(bytes.sublist(0, _frameBytes));
}

/// Grab a single reference frame (PNG bytes) via ffmpeg_kit for calibration.
Future<Uint8List> referenceFrameMobile(String videoPath, double atSec) async {
  final tmp = await getTemporaryDirectory();
  final out = '${tmp.path}/anya_ref_${DateTime.now().microsecondsSinceEpoch}.png';
  final s = await FFmpegKit.executeWithArguments([
    '-y',
    '-ss', atSec.toStringAsFixed(3),
    '-i', videoPath,
    '-frames:v', '1',
    '-vf', 'scale=$_w:$_h',
    '-sws_flags', 'bilinear',
    out,
  ]);
  final f = File(out);
  if (!ReturnCode.isSuccess(await s.getReturnCode()) || !f.existsSync()) {
    throw StateError('ffmpeg_kit reference-frame extraction failed');
  }
  final bytes = await f.readAsBytes();
  await _tryDelete(f);
  return bytes;
}

Future<void> _tryDelete(File f) async {
  try {
    await f.delete();
  } catch (_) {}
}
