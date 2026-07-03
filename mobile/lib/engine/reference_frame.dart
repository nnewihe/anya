import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

import 'config.dart';
import 'ffmpeg_mobile.dart';
import 'platform.dart';

/// Extract a single representative frame (default ~5s in), scaled to the
/// analysis resolution, as PNG bytes — used to drive court-corner calibration.
/// Desktop uses the ffmpeg CLI; mobile uses ffmpeg_kit.
Future<Uint8List> extractReferenceFrame(String videoPath,
    {double atSec = 5.0}) async {
  if (isMobilePlatform) return referenceFrameMobile(videoPath, atSec);

  final tmp = await getTemporaryDirectory();
  final out = '${tmp.path}/anya_refframe_${DateTime.now().microsecondsSinceEpoch}.png';
  final r = await Process.run('ffmpeg', [
    '-y',
    '-ss', atSec.toStringAsFixed(3),
    '-i', videoPath,
    '-frames:v', '1',
    '-vf', 'scale=${EngineConfig.analysisWidth}:${EngineConfig.analysisHeight}',
    '-sws_flags', 'bilinear',
    out,
  ]);
  final f = File(out);
  if (r.exitCode != 0 || !f.existsSync()) {
    throw StateError('failed to extract reference frame: ${r.stderr}');
  }
  final bytes = await f.readAsBytes();
  await f.delete().catchError((_) => f);
  return bytes;
}
