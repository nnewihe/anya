// YouTubeUploader.swift
//
// Upload a reel (or the whole recording) to the user's YouTube channel.
//
//   1. Sign in with Google (GoogleSignIn SDK), asking only for the
//      youtube.upload scope. The SDK keeps the tokens in the keychain; this
//      app never stores or logs them.
//   2. Open a RESUMABLE upload session (YouTube Data API v3). The session URL
//      itself authorises the upload, for about a week, so it is saved in the
//      match manifest.
//   3. PUT the file through a BACKGROUND URLSession: it keeps going when the
//      app is closed, and if it is cut off, `resume` asks YouTube how much
//      arrived and sends the rest.
//
// Needs an iOS OAuth client for com.build2launch.anyatennis: put its client
// ID in Info.plist as GIDClientID and its reversed client ID as a URL scheme.
// Until then `isConfigured` is false and the app hides the button.
import Foundation
import UIKit
#if canImport(GoogleSignIn)
import GoogleSignIn
#endif

enum YouTubeError: LocalizedError {
    case notConfigured, signInFailed(String), sessionRefused(Int), noSessionURL
    var errorDescription: String? {
        switch self {
        case .notConfigured: "YouTube upload isn't set up in this build."
        case .signInFailed(let message): "Google sign-in failed: \(message)"
        case .sessionRefused(let code): "YouTube refused the upload (HTTP \(code))."
        case .noSessionURL: "YouTube did not return an upload address."
        }
    }
}

final class YouTubeUploader: NSObject, URLSessionDataDelegate {
    static let shared = YouTubeUploader()
    static let uploadScope = "https://www.googleapis.com/auth/youtube.upload"
    private static let sessionEndpoint = URL(string:
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status")!

    /// Called on the main queue when an upload finishes: (match id, video id or nil, error).
    var onFinished: ((String, String?, Error?) -> Void)?
    /// Set by the app delegate when iOS wakes the app for background events.
    var backgroundCompletion: (() -> Void)?

    private var responses: [Int: Data] = [:]
    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.background(withIdentifier: "com.build2launch.anyatennis.youtube")
        configuration.sessionSendsLaunchEvents = true
        configuration.isDiscretionary = false
        return URLSession(configuration: configuration, delegate: self, delegateQueue: .main)
    }()

    static var isConfigured: Bool {
        #if canImport(GoogleSignIn)
        let id = Bundle.main.object(forInfoDictionaryKey: "GIDClientID") as? String ?? ""
        return id.hasSuffix(".apps.googleusercontent.com")
        #else
        return false
        #endif
    }

    /// Recreate the background session at launch so iOS can deliver events.
    func reconnect() { _ = session }

    // ── sign-in ───────────────────────────────────────────────────────────

    @MainActor
    static func accessToken(presenting viewController: UIViewController) async throws -> String {
        #if canImport(GoogleSignIn)
        guard isConfigured else { throw YouTubeError.notConfigured }
        let instance = GIDSignIn.sharedInstance
        var user = try? await instance.restorePreviousSignIn()
        if user == nil || !(user!.grantedScopes ?? []).contains(uploadScope) {
            do {
                user = try await instance.signIn(withPresenting: viewController, hint: nil,
                                                 additionalScopes: [uploadScope]).user
            } catch {
                throw YouTubeError.signInFailed(error.localizedDescription)
            }
        }
        let refreshed = try await user!.refreshTokensIfNeeded()
        return refreshed.accessToken.tokenString
        #else
        throw YouTubeError.notConfigured
        #endif
    }

    // ── upload ────────────────────────────────────────────────────────────

    /// Open a resumable session for `file`. Returns the session URL.
    func openSession(file: URL, title: String, description: String, privacy: String,
                     accessToken: String) async throws -> URL {
        let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        var request = URLRequest(url: Self.sessionEndpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json; charset=UTF-8", forHTTPHeaderField: "Content-Type")
        request.setValue(String(size), forHTTPHeaderField: "X-Upload-Content-Length")
        request.setValue("video/quicktime", forHTTPHeaderField: "X-Upload-Content-Type")
        let metadata: [String: Any] = [
            "snippet": ["title": String(title.prefix(100)), "description": String(description.prefix(4900)),
                        "categoryId": "17"],                           // Sports
            "status": ["privacyStatus": privacy, "selfDeclaredMadeForKids": false],
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: metadata)
        let (_, response) = try await URLSession.shared.data(for: request)
        let http = response as? HTTPURLResponse
        guard http?.statusCode == 200 else { throw YouTubeError.sessionRefused(http?.statusCode ?? -1) }
        guard let location = http?.value(forHTTPHeaderField: "Location"), let url = URL(string: location),
              Self.isSessionURL(url)
        else { throw YouTubeError.noSessionURL }
        return url
    }

    /// A YouTube resumable-upload address, and nothing else: the file is
    /// only ever sent to an address that passes this (also when the address
    /// comes back from the manifest on resume).
    static func isSessionURL(_ url: URL) -> Bool {
        guard url.scheme == "https", url.host == "www.googleapis.com",
              url.path.hasPrefix("/upload/youtube/v3/videos"),
              let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
        else { return false }
        return query.contains { $0.name == "upload_id" && !($0.value ?? "").isEmpty }
    }

    @MainActor
    static func signOut() async {
        #if canImport(GoogleSignIn)
        try? await GIDSignIn.sharedInstance.disconnect()
        #endif
    }

    private static func remainderFile(_ matchID: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("youtube-\(UUID(uuidString: matchID)?.uuidString ?? "x").part")
    }

    /// Send `file` (from byte `offset`) to the session, in the background.
    func send(file: URL, from offset: Int64 = 0, sessionURL: URL, matchID: String) throws {
        guard Self.isSessionURL(sessionURL) else { throw YouTubeError.noSessionURL }
        let size = Int64(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        var body = file
        if offset > 0 {
            // A background upload must come from a file: write the remainder out.
            body = Self.remainderFile(matchID)
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(offset))
            FileManager.default.createFile(atPath: body.path, contents: nil)
            let out = try FileHandle(forWritingTo: body)
            defer { try? out.close() }
            while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { try out.write(contentsOf: chunk) }
        }
        var request = URLRequest(url: sessionURL)
        request.httpMethod = "PUT"
        request.setValue("video/quicktime", forHTTPHeaderField: "Content-Type")
        if offset > 0 { request.setValue("bytes \(offset)-\(size - 1)/\(size)", forHTTPHeaderField: "Content-Range") }
        let task = session.uploadTask(with: request, fromFile: body)
        task.taskDescription = matchID
        task.resume()
    }

    /// Continue an interrupted upload: ask how much YouTube has, send the rest.
    func resume(file: URL, sessionURL: URL, matchID: String) async throws {
        guard Self.isSessionURL(sessionURL) else { throw YouTubeError.noSessionURL }
        let size = Int64(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        var request = URLRequest(url: sessionURL)
        request.httpMethod = "PUT"
        request.setValue("bytes */\(size)", forHTTPHeaderField: "Content-Range")
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as? HTTPURLResponse
        if http?.statusCode == 200 || http?.statusCode == 201 {          // it had already finished
            finish(matchID: matchID, data: data, error: nil)
            return
        }
        guard http?.statusCode == 308 else { throw YouTubeError.sessionRefused(http?.statusCode ?? -1) }
        // "Range: bytes=0-12345" = received through byte 12345.
        let received = http?.value(forHTTPHeaderField: "Range")
            .flatMap { $0.split(separator: "-").last }.flatMap { Int64($0) }.map { $0 + 1 } ?? 0
        guard (0..<size).contains(received) else { throw YouTubeError.sessionRefused(308) }
        try send(file: file, from: received, sessionURL: sessionURL, matchID: matchID)
    }

    // ── URLSession delegate ───────────────────────────────────────────────

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        responses[dataTask.taskIdentifier, default: Data()].append(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let data = responses.removeValue(forKey: task.taskIdentifier) ?? Data()
        let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        let failure = error ?? ((200...201).contains(status) ? nil : YouTubeError.sessionRefused(status))
        finish(matchID: task.taskDescription ?? "", data: data, error: failure)
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        backgroundCompletion?()
        backgroundCompletion = nil
    }

    private func finish(matchID: String, data: Data, error: Error?) {
        try? FileManager.default.removeItem(at: Self.remainderFile(matchID))
        let videoID = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["id"] as? String
        onFinished?(matchID, videoID, videoID == nil ? (error ?? YouTubeError.noSessionURL) : nil)
    }
}
