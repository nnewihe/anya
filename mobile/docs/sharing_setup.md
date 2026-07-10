# Reel sharing setup

The analysis screen offers two ways to share a generated rally reel:

| Action | Works out of the box? |
| --- | --- |
| **Save to Gallery** | ✅ Yes — nothing to configure. |
| **Upload to YouTube** | ⚠️ Needs a one-time Google OAuth setup (below). |

## Save to Gallery

Fully wired via the `gal` package. The permission strings are already in place
(`NSPhotoLibraryAddUsageDescription` in the iOS/macOS `Info.plist`, and the
`READ_MEDIA_VIDEO` / storage permissions in `AndroidManifest.xml`). The OS
prompts for Photos/Gallery access the first time; the reel is saved to a
"Rally Predictor" album.

## Upload to YouTube — one-time OAuth setup

Uploading uses the YouTube Data API v3 resumable-upload flow, authorized with a
Google account via `google_sign_in`. The OAuth client IDs are tied to **your
own** Google Cloud project, so they can't be committed here — the code ships
with placeholders and a gate (`kYouTubeUploadConfigured = false`) that makes the
Upload button explain this step until you finish it.

### 1. Google Cloud project

1. Create (or pick) a project at <https://console.cloud.google.com>.
2. **APIs & Services → Library →** enable **YouTube Data API v3**.
3. **APIs & Services → OAuth consent screen:** external, add the scope
   `https://www.googleapis.com/auth/youtube.upload`, and add your Google
   account as a **Test user** (an unverified app is limited to test users).

### 2. Create OAuth client IDs

**APIs & Services → Credentials → Create Credentials → OAuth client ID**, once
per platform you ship:

- **iOS** — bundle ID `com.example.rallyPredictor`
- **macOS** — bundle ID `com.example.rallyPredictor`
- **Android** — package `com.example.rally_predictor` + your signing-key SHA-1
  (`keytool -list -v -keystore ~/.android/debug.keystore -alias androiddebugkey
  -storepass android` for debug builds)

### 3. Fill in the placeholders

**iOS** — `ios/Runner/Info.plist`: replace both `YOUR_IOS_OAUTH_CLIENT_ID`
occurrences (the `GIDClientID` keeps the `.apps.googleusercontent.com` suffix;
the URL scheme uses the **reversed** form `com.googleusercontent.apps.<id>`).

**macOS** — `macos/Runner/Info.plist`: same, with the macOS client ID.

**Android** — nothing in the app: the console matches the OAuth client by
package name + SHA-1. Just make sure the SHA-1 you registered matches the
keystore you build with.

### 4. Flip the gate

In `lib/services/youtube_config.dart` set:

```dart
const bool kYouTubeUploadConfigured = true;
```

Rebuild. Tapping **Upload to YouTube** now signs in and uploads the reel as a
**private** video (title `"<clip> — Rally Reel"`). Single-attempt upload — a
failed transfer is retried from scratch, which is fine for these short clips.

### Notes

- Videos upload as **private** by default; change `privacyStatus` in
  `lib/services/youtube_upload.dart` if you want unlisted/public.
- Until Google verifies the app, only the **Test users** you added can
  authorize it (a "Google hasn't verified this app" warning is expected —
  proceed via *Advanced*).
