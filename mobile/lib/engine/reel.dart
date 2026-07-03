import 'dart:io';

import 'ffmpeg_mobile.dart';
import 'platform.dart';

/// Cut the detected segments from the source video and concatenate them into a
/// highlight reel. Port of utilities.create_highlights_ffmpeg. Shared merge /
/// pre-roll logic; the cut+concat step runs via ffmpeg_kit on mobile and the
/// system ffmpeg binary on desktop.
Future<bool> createHighlightReel({
  required String videoPath,
  required List<(double, double)> segments,
  required String outputPath,
  double preRoll = 0.0,
  double mergeGapSec = 3.0,
}) async {
  if (segments.isEmpty) return false;

  // Pre-roll, drop zero-length, sort, then merge close gaps.
  final extended = [
    for (final (s, e) in segments)
      if (e > s) (start: s < preRoll ? 0.0 : s - preRoll, end: e)
  ];
  if (extended.isEmpty) return false;
  extended.sort((a, b) => a.start.compareTo(b.start));
  final valid = <({double start, double end})>[extended.first];
  for (final seg in extended.skip(1)) {
    final last = valid.last;
    if (seg.start - last.end <= mergeGapSec) {
      valid[valid.length - 1] =
          (start: last.start, end: seg.end > last.end ? seg.end : last.end);
    } else {
      valid.add(seg);
    }
  }

  if (isMobilePlatform) {
    return reelCutMobile(
      videoPath: videoPath,
      valid: [for (final v in valid) (v.start, v.end)],
      outputPath: outputPath,
    );
  }

  final tmpDir = await Directory.systemTemp.createTemp('anya_highlights_');
  try {
    final segFiles = <String>[];
    for (var i = 0; i < valid.length; i++) {
      final seg = valid[i];
      final segPath = '${tmpDir.path}/seg_${i.toString().padLeft(4, '0')}.mp4';
      final r = await Process.run('ffmpeg', [
        '-y',
        '-ss', seg.start.toStringAsFixed(3),
        '-to', seg.end.toStringAsFixed(3),
        '-i', videoPath,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-vsync', 'cfr',
        segPath,
      ]);
      if (r.exitCode == 0) segFiles.add(segPath);
    }
    if (segFiles.isEmpty) return false;

    final concatList = File('${tmpDir.path}/concat.txt');
    await concatList
        .writeAsString(segFiles.map((f) => "file '$f'").join('\n'));

    final r = await Process.run('ffmpeg', [
      '-y',
      '-f', 'concat', '-safe', '0',
      '-i', concatList.path,
      '-c', 'copy',
      outputPath,
    ]);
    return r.exitCode == 0;
  } finally {
    await tmpDir.delete(recursive: true);
  }
}
