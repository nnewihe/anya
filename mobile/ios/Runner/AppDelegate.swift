import BackgroundTasks
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private static let processingTaskId = "com.build2launch.anya_tennis.analysis-processing"
  private var backgroundTaskId: UIBackgroundTaskIdentifier = .invalid

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    registerProcessingTask()
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    // Background-analysis support channel: exposes UIApplication's
    // beginBackgroundTask/endBackgroundTask assertion (extends the grace
    // window iOS grants right after backgrounding, per DESIGN.md §6.2) and
    // lets Dart nudge the OS to (re-)schedule the BGProcessingTask below.
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "AnyaBackgroundTaskChannel") {
      let channel = FlutterMethodChannel(
        name: "anya_tennis/background_task", binaryMessenger: registrar.messenger())
      channel.setMethodCallHandler { [weak self] call, result in
        guard let self = self else { return }
        switch call.method {
        case "beginBackgroundTask":
          self.beginBackgroundTask()
          result(nil)
        case "endBackgroundTask":
          self.endBackgroundTask()
          result(nil)
        case "scheduleProcessingTask":
          self.scheduleProcessingTask()
          result(nil)
        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }
  }

  // MARK: - UIApplication.beginBackgroundTask assertion
  //
  // Extends the grace window iOS grants after the app backgrounds (a
  // best-effort handful of extra seconds — iOS makes no time guarantee) so
  // an in-flight analysis frame/window can finish cleanly instead of being
  // suspended mid-write. Not unbounded: the expiration handler fires
  // regardless once the OS decides time is up, and we end the assertion
  // ourselves at that point so the app doesn't get penalized for overrunning it.
  private func beginBackgroundTask() {
    guard backgroundTaskId == .invalid else { return }
    backgroundTaskId = UIApplication.shared.beginBackgroundTask(withName: "anya-analysis") {
      [weak self] in
      self?.endBackgroundTask()
    }
  }

  private func endBackgroundTask() {
    guard backgroundTaskId != .invalid else { return }
    UIApplication.shared.endBackgroundTask(backgroundTaskId)
    backgroundTaskId = .invalid
  }

  // MARK: - BGProcessingTask
  //
  // Registered identifier the system can schedule under favorable conditions
  // (charging / idle) for jobs longer than the background-task assertion
  // above can cover. This registration gives the OS a well-behaved task to
  // schedule and complete, satisfying the BGTaskSchedulerPermittedIdentifiers
  // contract (Info.plist) and the "system schedules when conditions allow"
  // half of DESIGN.md §6.2.
  //
  // NOT implemented here: resuming a SPECIFIC in-flight Dart analysis job
  // from a cold BGProcessingTask launch. That needs job-state persistence on
  // the Dart side (which video, how far progress got, the partial reel path)
  // so a relaunched engine isolate knows where to pick up — out of scope for
  // this pass. The handler below re-arms the next scheduling opportunity and
  // completes; it does not currently invoke the Flutter engine.
  private func registerProcessingTask() {
    BGTaskScheduler.shared.register(
      forTaskWithIdentifier: AppDelegate.processingTaskId, using: nil
    ) { task in
      self.handleProcessingTask(task: task as! BGProcessingTask)
    }
  }

  private func handleProcessingTask(task: BGProcessingTask) {
    scheduleProcessingTask()  // keep the chain alive for the next opportunity
    task.setTaskCompleted(success: true)
  }

  private func scheduleProcessingTask() {
    let request = BGProcessingTaskRequest(identifier: AppDelegate.processingTaskId)
    request.requiresNetworkConnectivity = false
    request.requiresExternalPower = false
    do {
      try BGTaskScheduler.shared.submit(request)
    } catch {
      // Scheduling can fail (e.g. simulator without background-task support,
      // or a request already pending) — non-fatal, best-effort by design.
    }
  }
}
