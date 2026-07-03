import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;

/// True on Android / iOS, where video decode + reel run through ffmpeg_kit.
/// Desktop (macOS/Windows/Linux) shells out to the system `ffmpeg` binary.
bool get isMobilePlatform =>
    !kIsWeb && (Platform.isAndroid || Platform.isIOS);
