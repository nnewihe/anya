/// App configuration. Override the backend URL at build time with:
///   flutter run --dart-define=API_BASE_URL=https://api.example.com
class AppConfig {
  /// REST base, e.g. http://10.0.2.2:8000 (Android emulator → host machine)
  /// or http://localhost:8000 (iOS simulator).
  static const String apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://44.203.32.208:8000',
  );

  /// WebSocket base derived from the REST base (http→ws, https→wss).
  static String get wsBaseUrl {
    final u = Uri.parse(apiBaseUrl);
    final scheme = u.scheme == 'https' ? 'wss' : 'ws';
    return u.replace(scheme: scheme).toString();
  }
}
