# Anya Tennis on TestFlight and the App Store

The app record is `com.build2launch.anyatennis`, team `696S9GCN96` (set in `project.yml`).
Build 0.1.0 (1) was uploaded on 2026-11-26 with Xcode 27.2 / the iOS 27.2 SDK, minimum iOS 16.0.

## Uploading a build

Apple requires the current SDK. Uploads from Xcode 16 / iOS 18.5 were refused; the iOS 27 SDK is required from April 2027.

1. Bump `CFBundleVersion` in `project.yml`. Every upload needs a higher build number than the last; `CFBundleShortVersionString` changes only for a new store version. Then run `xcodegen`.
2. Archive, from `ios/AnyaTennis`:

   ```bash
   xcodebuild -project AnyaTennis.xcodeproj -scheme AnyaTennis -configuration Release -destination 'generic/platform=iOS' -archivePath build/AnyaTennis.xcarchive -allowProvisioningUpdates archive
   ```

3. Upload:

   ```bash
   xcodebuild -exportArchive -archivePath build/AnyaTennis.xcarchive -exportOptionsPlist tools/ExportOptions.plist -exportPath build/export -allowProvisioningUpdates
   ```

App Store Connect processes a build for 10–30 minutes, then it appears under TestFlight. Export compliance is already answered: `ITSAppUsesNonExemptEncryption = false`.

## TestFlight

**Internal testers** (users on the App Store Connect team, up to 100):
- Go to TestFlight → Internal Testing → **+**, and add the testers.
- Builds reach them with no review.
- Testers install the TestFlight app and accept the email invite.

**External testers** (any email, or a public link, up to 10,000):
- Create a group under External Testing and add the build.
- Fill in **Test Information** with the text below.
- The first build of each version goes through Beta App Review, usually about a day.

**Beta App Description**

> Anya Tennis turns a phone on a tripod into a match recorder. Put the phone behind the baseline, tap the four corners of the singles court, and record. While you play, the app finds where every point starts and ends, and when you tap Stop it cuts a highlight reel of just the points, with the walking around and ball-collecting removed. Everything runs on the phone: no account, no upload, no server.

**What to Test**

> Record a real match or practice set, ideally 15 minutes or more. Use a tripod or fence mount behind the baseline, high enough to see the whole court. Check that:
> - calibration accepts your four taps
> - the "analysis N s behind" counter keeps moving while you record
> - the reel after Stop contains your points and not the gaps between them
>
> Also try taking a call or leaving the app mid-recording, then resuming. Tell us your phone model and anything that looked wrong, especially points that were missed or cut short.

**Feedback email:** smarttenniswithandy@gmail.com

## Review notes

Use these for both Beta App Review and App Review.

> Anya Tennis records a tennis match from a phone on a tripod and makes a highlight reel of the points. It needs a tennis court to show its main result. Without one, reviewers can still try the full flow:
> 1. Tap **New match** and allow the camera.
> 2. Tap any four points on the preview as the "court corners", for example the corners of a table or a room.
> 3. Tap **Stop** after a minute. The app analyses the recording, and if it finds no points it says so ("No points were found in this recording") and keeps the full recording, which can be saved to Photos.
>
> The app needs at least 10 GB of free storage to start recording, because a match is about 15 GB an hour at 4K.
>
> There is no account or login. The optional YouTube upload uses Google Sign-In only to authorize uploading to the user's own channel; it is not a way of logging in to the app, so Sign in with Apple does not apply. All analysis runs on the device, and the app has no server.

## Store listing (App Store tab)

| Field | Text |
|---|---|
| Name | Anya Tennis |
| Subtitle (30) | Match recorder & point reels |
| Category | Sports (secondary: Photo & Video) |
| Age rating | 4+ (answer "None" to every content question) |
| Price | your choice; the app has no in-app purchases |
| Support URL | https://anyatennis.com |
| Marketing URL | https://anyatennis.com |
| Privacy Policy URL | https://nnewihe.github.io/anya/privacy.html (the iPhone section is `#iphone`) |
| Copyright | 2026 ICON Tennis & Pickleball |

**Promotional text (170)**

> Record your match from a tripod and get a reel of every point the moment you tap Stop. No account, no upload: it all runs on your iPhone.

**Description**

> Anya Tennis records your tennis match and cuts out everything that isn't tennis.
>
> Put your iPhone on a tripod behind the baseline, tap the four corners of the singles court, and press record. While you play, Anya watches both players, recognises serves at either end, and spots when each point is over. Tap Stop and your highlight reel is ready in moments: every point, in order, without the walking, towel breaks and ball-collecting in between.
>
> • Records in 4K, with the full match kept alongside the reel
> • Finds point starts from serves at both ends of the court, and point ends from how the players move
> • Analysis runs live while you record, so the reel is ready right after Stop
> • Interrupted by a call? The recording pauses safely and resumes where it left off
> • Save to Photos, share, or upload to your own YouTube channel
> • Private by design: no account, no server. Your video never leaves your phone unless you send it somewhere.
>
> Works on iPhone X and later with iOS 16 or newer. It runs fastest on iPhone XS and later, which have a Neural Engine.

**Keywords (100)**

> tennis,match,recorder,highlights,reel,points,serve,video,tripod,coach,practice,rally,court,camera

**Screenshots:**
- Required: 6.9" (1320×2868 or 2868×1320) and 6.5" (1284×2778 or 2778×1284).
- The app is landscape, so take landscape screenshots of calibration, recording with the status bar, Finish, and the reel.
- Capture from a phone, or from the iPhone 16 Pro Max / 16 Plus simulators, with `xcrun simctl io booted screenshot`.

## App Privacy (App Store Connect → App Privacy)

- **Data collection:** "No, we do not collect data from this app."
- This matches `AnyaTennis/Resources/PrivacyInfo.xcprivacy`: no tracking, no collected data types.
- A YouTube upload goes from the device straight to Google at the user's request, and the developer receives nothing. Under Apple's definitions that is not collection by the developer.
