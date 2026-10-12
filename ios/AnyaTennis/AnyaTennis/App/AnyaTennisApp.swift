// AnyaTennisApp.swift
//
// Record a tennis match; get the highlight reel seconds after Stop.
//
// Screens, in the order a match goes through them:
//   MatchListView     home: past matches, "New match", interrupted matches
//   CalibrateView     camera preview; tap the four singles corners
//   RecordView        recording, with live analysis status; Stop
//   (PausedView)      after a phone call / backgrounding: resume or finish
//   FinishView        analysis catches up and the reel is cut; match info
//                     is typed here meanwhile
//   MatchDetailView   the reel: play, save to Photos, share, YouTube
import AVFoundation
import SwiftUI
#if canImport(GoogleSignIn)
import GoogleSignIn
#endif

@main
struct AnyaTennisApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .preferredColorScheme(.dark)
                #if canImport(GoogleSignIn)
                .onOpenURL { GIDSignIn.sharedInstance.handle($0) }
                #endif
                .task { await model.launch() }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    /// Landscape-right only: the camera sensor's own orientation (CameraSession).
    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        .landscapeRight
    }

    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        YouTubeUploader.shared.backgroundCompletion = completionHandler
        YouTubeUploader.shared.reconnect()
    }
}

/// Which screen is up, and the match it is about.
@MainActor
@Observable
final class AppModel {
    enum Screen: Equatable {
        case list
        case calibrate
        case record
        case paused
        case finish
        case detail(String)
    }

    var screen: Screen = .list
    var matches: [MatchManifest] = []
    var current: MatchController?
    var error: String?
    let camera = CameraSession()

    func launch() async {
        YouTubeUploader.shared.reconnect()
        YouTubeUploader.shared.onFinished = { [weak self] matchID, videoID, error in
            self?.uploadFinished(matchID: matchID, videoID: videoID, error: error)
        }
        var all = MatchStore.all()
        for i in all.indices { await MatchController.recoverIfInterrupted(&all[i]) }
        matches = MatchStore.all()
        #if DEBUG
        if let replay = ReplayRequest.fromLaunchArguments() { await runReplay(replay) }
        if UserDefaults.standard.bool(forKey: "AnyaFinishFailed") {
            for match in matches where match.state == .failed || match.state == .paused {
                print("[replay] finishing \(match.id) (\(match.state.rawValue))")
                await finish(match)
                print("[replay] state \(current?.manifest.state.rawValue ?? "?"); reel \(String(describing: current?.manifest.reel)); "
                      + "error \(current?.manifest.lastError ?? "none")")
            }
        }
        #endif
        NotificationCenter.default.addObserver(forName: AVCaptureSession.wasInterruptedNotification,
                                               object: camera.session, queue: .main) { [weak self] _ in
            Task { @MainActor in await self?.interrupted() }
        }
        NotificationCenter.default.addObserver(forName: AVCaptureSession.runtimeErrorNotification,
                                               object: camera.session, queue: .main) { [weak self] _ in
            Task { @MainActor in await self?.interrupted() }
        }
    }

    func refresh() { matches = MatchStore.all() }

    // ── moving between screens ────────────────────────────────────────────

    func newMatch() async {
        guard await CameraSession.requestPermissions() else {
            error = CameraSession.SetupError.cameraPermission.localizedDescription
            return
        }
        do {
            try camera.configure()
            camera.start()
            screen = .calibrate
        } catch {
            self.error = error.localizedDescription
        }
    }

    func calibrated(_ controller: MatchController) async {
        current = controller
        await record()
    }

    func record() async {
        guard let current else { return }
        current.onRecordingFailed = { [weak self] _ in
            self?.screen = .paused
            self?.refresh()
        }
        do {
            try await current.startRecording()
            screen = .record
        } catch {
            self.error = error.localizedDescription
            refresh()
            screen = .list   // the match stays listed (calibrated) and can be recorded later
        }
    }

    /// Resume recording an interrupted (paused) match.
    func resume(_ manifest: MatchManifest) async {
        if current?.manifest.id != manifest.id { current = MatchController(manifest: manifest, camera: camera) }
        guard await CameraSession.requestPermissions() else { return }
        do {
            try camera.configure()
            camera.start()
            await record()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func stop() async {
        guard let current else { return }
        screen = .finish
        await current.stop()
        camera.stop()
        refresh()
    }

    /// Finish a paused or failed match without recording more.
    func finish(_ manifest: MatchManifest) async {
        if current?.manifest.id != manifest.id { current = MatchController(manifest: manifest, camera: camera) }
        screen = .finish
        current?.update { $0.state = .finishing }
        await current?.finish()
        refresh()
    }

    func showDetail(_ id: String) {
        if current?.manifest.id != id, let manifest = try? MatchStore.load(id) {
            current = MatchController(manifest: manifest, camera: camera)
        }
        screen = .detail(id)
    }

    func home() {
        if current?.isRecording != true { current = nil; camera.stop() }
        refresh()
        screen = .list
    }

    func delete(_ id: String) {
        try? MatchStore.delete(id)
        refresh()
    }

    #if DEBUG
    /// A whole match from a file, through the real recording path (FileReplay).
    private func runReplay(_ replay: ReplayRequest) async {
        do {
            let format = try await FileReplay.format(of: replay.url)
            let first = try await FileReplay.calibrationFrame(of: replay.url)
            guard let near = FrameScaler(band: .init(x: 0, y: 0, width: 2, height: 2)).near(first) else { return }
            let controller = try MatchController.create(corners: replay.corners, calibrationFrame: near, camera: camera,
                                                        sourceWidth: format.width, sourceHeight: format.height,
                                                        sourceFPS: format.fps)
            current = controller
            try await controller.startRecording()
            screen = .record
            print("[replay] recording \(replay.url.lastPathComponent) at \(replay.speed)x")
            try await FileReplay.play(replay.url, into: camera, speed: replay.speed)
            print("[replay] file done; stopping")
            await stop()
            print("[replay] state \(controller.manifest.state.rawValue); reel \(String(describing: controller.manifest.reel)); "
                  + "error \(controller.manifest.lastError ?? "none")")
        } catch {
            print("[replay] failed: \(error)")
            self.error = error.localizedDescription
        }
    }
    #endif

    private func interrupted() async {
        // Only while recording: during Stop the part is already ending.
        guard let current, current.isRecording, screen == .record else { return }
        await current.pauseRecording()
        screen = .paused
        refresh()
    }

    private func uploadFinished(matchID: String, videoID: String?, error: Error?) {
        guard var manifest = try? MatchStore.load(matchID) else { return }
        manifest.upload.videoID = videoID
        if videoID != nil { manifest.upload.sessionURL = nil }
        manifest.lastError = error?.localizedDescription
        if current?.manifest.id == matchID { current?.update { $0 = manifest } } else { try? MatchStore.save(manifest) }
        refresh()
    }
}
