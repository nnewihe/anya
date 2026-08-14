import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show MethodChannel;
import 'package:flutter_foreground_task/flutter_foreground_task.dart';

/// iOS-only native channel (see ios/Runner/AppDelegate.swift): takes a
/// UIApplication.beginBackgroundTask assertion while analysis is in flight
/// (extends the grace window iOS grants right after backgrounding — best
/// effort, no time guarantee) and nudges the OS to schedule the
/// BGProcessingTask for jobs too long for that window (DESIGN.md §6.2).
const _bgTaskChannel = MethodChannel('anya_tennis/background_task');

/// Keeps on-device analysis alive when the app is not in the foreground, and
/// surfaces progress + a completion notification.
///
/// Mechanism per platform:
///  * **Android** — a foreground service with an ongoing "Analysing your
///    match — NN%" notification. The service keeps the app process running
///    (and holds a wake-lock) so the engine keeps working while backgrounded.
///  * **iOS** — best-effort: the plugin keeps a short background window and the
///    notification tells the user when the reel is ready. iOS does not permit
///    unbounded background CPU, so a long match may pause and resume on return.
///  * **Desktop / web** — a no-op. Desktop OSes don't suspend a running app, so
///    analysis continues normally; every call here short-circuits.
///
/// This wraps `flutter_foreground_task`; the engine itself is unchanged.
class BackgroundAnalysis {
  BackgroundAnalysis._();

  /// The plugin only has Android/iOS implementations. Guard every call so the
  /// macOS/desktop/web Flutter builds don't hit a missing platform channel.
  static bool get _supported =>
      !kIsWeb && (Platform.isAndroid || Platform.isIOS);

  static bool _initialized = false;

  static void _ensureInit() {
    if (_initialized) return;
    FlutterForegroundTask.init(
      androidNotificationOptions: AndroidNotificationOptions(
        channelId: 'anya_match_analysis',
        channelName: 'Match analysis',
        channelDescription:
            'Shows progress while Anya Tennis analyses your match.',
        onlyAlertOnce: true,
      ),
      iosNotificationOptions: const IOSNotificationOptions(
        showNotification: true,
        playSound: false,
      ),
      foregroundTaskOptions: ForegroundTaskOptions(
        // We drive the work from the app's main isolate; the service just keeps
        // the process alive, so it needs no repeating event callback.
        eventAction: ForegroundTaskEventAction.nothing(),
        allowWakeLock: true,
        allowWifiLock: false,
        autoRunOnBoot: false,
        allowAutoRestart: false,
      ),
    );
    _initialized = true;
  }

  /// Ask for notification permission (Android 13+/iOS). Safe to call each run.
  static Future<void> requestPermissions() async {
    if (!_supported) return;
    _ensureInit();
    final status = await FlutterForegroundTask.checkNotificationPermission();
    if (status != NotificationPermission.granted) {
      await FlutterForegroundTask.requestNotificationPermission();
    }
  }

  /// Start the foreground service with an initial "analysing" notification,
  /// and (iOS only) take the background-task assertion + arm the
  /// BGProcessingTask fallback.
  static Future<void> start() async {
    if (!_supported) return;
    _ensureInit();
    if (Platform.isIOS) {
      // Fire-and-forget: a missing/failed native channel call shouldn't block
      // analysis from starting.
      unawaited(_bgTaskChannel.invokeMethod('beginBackgroundTask').catchError((_) {}));
      unawaited(_bgTaskChannel.invokeMethod('scheduleProcessingTask').catchError((_) {}));
    }
    if (await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.startService(
      serviceId: 4287,
      serviceTypes: const [ForegroundServiceTypes.dataSync],
      notificationTitle: 'Analysing your match',
      notificationText: 'Starting…',
    );
  }

  /// Update the ongoing notification with the current progress.
  static Future<void> update(double fraction, String message) async {
    if (!_supported) return;
    if (!await FlutterForegroundTask.isRunningService) return;
    final pct = (fraction.clamp(0.0, 1.0) * 100).toStringAsFixed(0);
    await FlutterForegroundTask.updateService(
      notificationTitle: 'Analysing your match — $pct%',
      notificationText: message,
    );
  }

  /// Swap the notification to a completion message. The service is left running
  /// so the notification persists until the user reopens the app (at which
  /// point [stop] clears it).
  static Future<void> complete({required String message}) async {
    if (!_supported) return;
    if (!await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.updateService(
      notificationTitle: 'Your rally reel is ready',
      notificationText: message,
    );
  }

  /// Stop the foreground service and clear its notification.
  static Future<void> stop() async {
    if (!_supported) return;
    if (Platform.isIOS) {
      unawaited(_bgTaskChannel.invokeMethod('endBackgroundTask').catchError((_) {}));
    }
    if (!await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.stopService();
  }
}
