/// One-time setup gate for YouTube upload.
///
/// Uploading needs Google OAuth client IDs that are tied to your own Google
/// Cloud project — see `docs/sharing_setup.md`.  Until those are filled in
/// (the iOS/macOS Info.plist `GIDClientID` + URL scheme, and the Android SHA-1
/// registered in the console), flip this to `true`.  While it is `false` the
/// Upload-to-YouTube button explains the setup step instead of triggering a
/// confusing native OAuth error.
const bool kYouTubeUploadConfigured = false;
