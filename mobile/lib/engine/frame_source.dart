import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'config.dart';
import 'ffmpeg_mobile.dart';
import 'platform.dart';

/// Open the right [FrameSource] for the current platform: ffmpeg_kit on mobile,
/// the system ffmpeg binary on desktop.
Future<FrameSource> openFrameSource(String videoPath) async {
  if (isMobilePlatform) return FfmpegKitFrameSource.open(videoPath);
  return FfmpegFrameSource.open(videoPath);
}

/// Grab ONE 960×540 rgb24 frame at time [tSec] (random access — used by the
/// exclusion-zone scan, which samples ~50 frames across the whole video).
/// Returns null if the seek/decode fails (e.g. past end of stream).
Future<Uint8List?> grabAnalysisFrame(String videoPath, double tSec) async {
  if (isMobilePlatform) return grabAnalysisFrameMobile(videoPath, tSec);
  final r = await Process.run(
    'ffmpeg',
    [
      '-v', 'error',
      '-ss', tSec.toStringAsFixed(3),
      '-i', videoPath,
      '-frames:v', '1',
      '-vf', 'scale=${EngineConfig.analysisWidth}:${EngineConfig.analysisHeight}',
      '-sws_flags', 'bilinear',
      '-pix_fmt', 'rgb24',
      '-f', 'rawvideo', '-',
    ],
    stdoutEncoding: null, // raw bytes
  );
  if (r.exitCode != 0) return null;
  final bytes = r.stdout as List<int>;
  const want = EngineConfig.analysisWidth * EngineConfig.analysisHeight * 3;
  if (bytes.length < want) return null;
  return Uint8List.fromList(bytes.sublist(0, want));
}

/// Video metadata needed to convert frame indices to source-video time.
class VideoInfo {
  final double fps;
  final int totalFrames;
  final double durationSec;
  const VideoInfo(this.fps, this.totalFrames, this.durationSec);
}

/// Streams 960×540 rgb24 frames from a video. The desktop implementation shells
/// out to ffmpeg; on mobile this is replaced by an ffmpeg_kit- or native
/// (AVAssetReader / MediaCodec) backed source behind the same interface.
/// Decode is not the throughput bottleneck (see spikes/FINDINGS.md).
abstract class FrameSource {
  VideoInfo get info;

  /// One [EngineConfig.analysisWidth]×[EngineConfig.analysisHeight] rgb24 frame
  /// per event, in order.
  Stream<Uint8List> frames();

  Future<void> dispose();
}

const int _frameBytes =
    EngineConfig.analysisWidth * EngineConfig.analysisHeight * 3;

class FfmpegFrameSource implements FrameSource {
  final String videoPath;
  @override
  final VideoInfo info;
  Process? _proc;

  FfmpegFrameSource._(this.videoPath, this.info);

  /// Probe the video (ffprobe) and construct a source.
  static Future<FfmpegFrameSource> open(String videoPath) async {
    final info = await _probe(videoPath);
    return FfmpegFrameSource._(videoPath, info);
  }

  static Future<VideoInfo> _probe(String videoPath) async {
    final r = await Process.run('ffprobe', [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
      '-of', 'json',
      videoPath,
    ]);
    var fps = 30.0, total = 0, dur = 0.0;
    try {
      final j = jsonDecode(r.stdout as String) as Map<String, dynamic>;
      final s = (j['streams'] as List).first as Map<String, dynamic>;
      final rate = (s['r_frame_rate'] as String).split('/');
      fps = double.parse(rate[0]) / double.parse(rate[1]);
      total = int.tryParse('${s['nb_frames']}') ?? 0;
      dur = double.tryParse('${s['duration']}') ?? 0.0;
    } catch (_) {}
    if (fps <= 0 || fps > 300) fps = 30.0;
    if (dur <= 0 && total > 0) dur = total / fps;
    if (total <= 0 && dur > 0) total = (dur * fps).round();
    return VideoInfo(fps, total, dur);
  }

  @override
  Stream<Uint8List> frames() async* {
    final proc = await Process.start('ffmpeg', [
      '-v', 'error',
      '-i', videoPath,
      '-vf', 'scale=${EngineConfig.analysisWidth}:${EngineConfig.analysisHeight}',
      '-sws_flags', 'bilinear',
      '-pix_fmt', 'rgb24',
      '-f', 'rawvideo', '-',
    ]);
    _proc = proc;
    proc.stderr.drain<void>();

    final buf = BytesBuilder();
    await for (final chunk in proc.stdout) {
      buf.add(chunk);
      while (buf.length >= _frameBytes) {
        final all = buf.takeBytes();
        yield Uint8List.fromList(all.sublist(0, _frameBytes));
        if (all.length > _frameBytes) buf.add(all.sublist(_frameBytes));
      }
    }
  }

  @override
  Future<void> dispose() async {
    _proc?.kill();
    _proc = null;
  }
}
