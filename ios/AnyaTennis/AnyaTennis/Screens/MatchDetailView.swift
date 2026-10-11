// MatchDetailView.swift
//
// A finished match: watch the reel, save it (or the whole recording) to
// Photos, share it, or upload it to YouTube.
import AVKit
import SwiftUI

struct MatchDetailView: View {
    @Environment(AppModel.self) private var model
    let matchID: String
    @State private var player: AVPlayer?
    @State private var busy = ""
    @State private var info = MatchManifest.Info()
    @State private var privacy = "private"

    private var manifest: MatchManifest? { model.current?.manifest }
    private var reelURL: URL? { manifest?.reel.map { MatchStore.file($0.file, of: matchID) } }

    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            VStack {
                if let player { VideoPlayer(player: player).aspectRatio(16 / 9, contentMode: .fit) }
                if let reel = manifest?.reel {
                    Text("\(reel.points) points · \(RecordView.clock(reel.seconds)) · "
                         + "\(reel.nearServes + reel.farServes) serves found")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .overlay { if manifest?.reel == nil { Text("No reel: no points were found.").foregroundStyle(.secondary) } }
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Button { model.home() } label: { Label("Matches", systemImage: "chevron.left") }
                        Spacer()
                    }
                    MatchInfoForm(info: $info)
                    Button("Save info") { model.current?.updateInfo(info) }.buttonStyle(.bordered)
                    Divider()
                    actions
                    if !busy.isEmpty { ProgressView(busy) }
                    if let error = manifest?.lastError { Text(error).font(.footnote).foregroundStyle(.orange) }
                }
            }
            .frame(width: 380)
        }
        .padding()
        .onAppear {
            info = manifest?.info ?? .init()
            if let reelURL { player = AVPlayer(url: reelURL) }
        }
        .onDisappear { player?.pause() }
    }

    @ViewBuilder private var actions: some View {
        Button {
            Task { await run("Saving the reel…") {
                guard let reelURL else { return }
                try await PhotosSaver.save(videos: [reelURL])
                model.current?.update { $0.savedReelToPhotos = true }
            } }
        } label: { Label(manifest?.savedReelToPhotos == true ? "Reel saved to Photos" : "Save reel to Photos",
                         systemImage: "photo.on.rectangle") }
        Button {
            Task { await run("Saving the full recording…") {
                guard let manifest else { return }
                try await PhotosSaver.save(videos: manifest.parts.map { MatchStore.file($0.file, of: matchID) })
                model.current?.update { $0.savedRawToPhotos = true }
            } }
        } label: { Label(manifest?.savedRawToPhotos == true ? "Recording saved to Photos" : "Save full recording to Photos",
                         systemImage: "film.stack") }
        if let reelURL {
            ShareLink(item: reelURL) { Label("Share the reel", systemImage: "square.and.arrow.up") }
        }
        if YouTubeUploader.isConfigured {
            youtube
        } else {
            Label("YouTube upload: not set up in this build", systemImage: "play.rectangle")
                .foregroundStyle(.secondary).font(.footnote)
        }
        Button(role: .destructive) {
            model.delete(matchID)
            model.home()
        } label: { Label("Delete match", systemImage: "trash") }
    }

    @ViewBuilder private var youtube: some View {
        if let videoID = manifest?.upload.videoID, let url = URL(string: "https://youtu.be/\(videoID)") {
            Link(destination: url) { Label("On YouTube", systemImage: "play.rectangle.fill") }
        } else if manifest?.upload.sessionURL != nil {
            Button {
                Task { await run("Resuming the upload…") {
                    guard let manifest, let session = manifest.upload.sessionURL, let reelURL else { return }
                    try await YouTubeUploader.shared.resume(file: reelURL, sessionURL: session, matchID: matchID)
                } }
            } label: { Label("Uploading to YouTube… (tap to resume)", systemImage: "arrow.clockwise") }
        } else {
            Picker("Visibility", selection: $privacy) {
                Text("Private").tag("private")
                Text("Unlisted").tag("unlisted")
                Text("Public").tag("public")
            }
            .pickerStyle(.segmented)
            Button {
                Task { await run("Starting the upload…") { try await uploadReel() } }
            } label: { Label("Upload the reel to YouTube", systemImage: "play.rectangle") }
        }
        Button("Disconnect Google account") { Task { await YouTubeUploader.signOut() } }
            .font(.footnote)
    }

    private func uploadReel() async throws {
        guard let manifest, let reelURL,
              let window = UIApplication.shared.connectedScenes.compactMap({ ($0 as? UIWindowScene)?.keyWindow }).first,
              let root = window.rootViewController else { return }
        let token = try await YouTubeUploader.accessToken(presenting: root)
        let title = manifest.displayTitle
        let players = manifest.info.players.filter { !$0.isEmpty }.joined(separator: ", ")
        let description = [players, manifest.info.location, manifest.info.notes, "Highlights cut by Anya Tennis."]
            .filter { !$0.isEmpty }.joined(separator: "\n")
        let session = try await YouTubeUploader.shared.openSession(file: reelURL, title: title, description: description,
                                                                   privacy: privacy, accessToken: token)
        model.current?.update { $0.upload.sessionURL = session; $0.upload.privacy = privacy }
        try YouTubeUploader.shared.send(file: reelURL, sessionURL: session, matchID: matchID)
    }

    private func run(_ label: String, _ work: () async throws -> Void) async {
        busy = label
        defer { busy = "" }
        do {
            try await work()
            model.current?.update { $0.lastError = nil }
        } catch {
            model.current?.update { $0.lastError = error.localizedDescription }
        }
    }
}
