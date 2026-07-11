/// One-time setup gate for YouTube upload.
///
/// OAuth client IDs configured 2026-07 — separate clients per platform type
/// (iOS and macOS each need their own, even sharing bundle ID
/// com.example.rallyPredictor; see Info.plist GIDClientID in both platforms).
/// Android needs no app-side change; the console matches by package + SHA-1
/// registered separately.  See `docs/sharing_setup.md`.
const bool kYouTubeUploadConfigured = true;
