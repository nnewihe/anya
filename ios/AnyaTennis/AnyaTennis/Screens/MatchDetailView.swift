// MatchDetailView.swift
//
// A finished match: watch the reel, save it (or the whole recording) to
// Photos, share it, or upload it to YouTube.
import AVKit
import SwiftUI

struct MatchDetailView: View {
    @EnvironmentObject private var model: AppModel
    let matchID: String
    @State private var player: AVPlayer?
    @State private var busy = ""
    @State private var info = MatchManifest.Info()
    @State private var privacy = "private"

    private var manifest: MatchManifest? { model.current?.manifest }
    private var reelURL: URL? { manifest?.reel.map { MatchStore.file($0.file, of: matchID) } }

    @State private var confirmDelete = false
    @State private var savedInfo = false

    var body: some View {
        HStack(alignment: .top, spacing: 22) {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    CircleIconButton(systemImage: "chevron.left") { model.showMatches() }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(manifest?.displayTitle ?? "Match").font(Theme.display(24, .heavy)).lineLimit(1)
                        if let date = manifest?.createdAt, manifest?.info.title.isEmpty == false {
                            Text(date.formatted(date: .abbreviated, time: .shortened))
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                }
                ZStack {
                    Theme.card
                    if let player {
                        VideoPlayer(player: player)
                    } else {
                        VStack(spacing: 10) {
                            Image(systemName: "film").font(.system(size: 34)).foregroundStyle(.tertiary)
                            Text("No reel: no points were found.").foregroundStyle(.secondary)
                        }
                    }
                }
                .aspectRatio(16 / 9, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous).stroke(Theme.stroke))
                if let reel = manifest?.reel {
                    HStack(spacing: 26) {
                        StatChip(value: "\(reel.points)", label: reel.points == 1 ? "point" : "points", systemImage: "tennisball")
                        StatChip(value: RecordView.clock(reel.seconds), label: "reel", systemImage: "film")
                        StatChip(value: "\(reel.nearServes + reel.farServes)", label: "serves found", systemImage: "figure.tennis")
                    }
                }
            }
            .frame(maxWidth: .infinity)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 12) {
                        SectionLabel("Match info")
                        MatchInfoForm(info: $info)
                        Button {
                            model.current?.updateInfo(info)
                            savedInfo = true
                        } label: { Label(savedInfo ? "Saved" : "Save info", systemImage: savedInfo ? "checkmark" : "square.and.arrow.down") }
                            .buttonStyle(.secondary)
                    }
                    .card(padding: 16)
                    VStack(alignment: .leading, spacing: 8) {
                        SectionLabel("Keep and share")
                        actions
                        if !busy.isEmpty { ProgressView(busy).padding(.top, 4) }
                        if let error = manifest?.lastError { Text(error).font(.footnote).foregroundStyle(Theme.warning) }
                    }
                    .card(padding: 16)
                    Button(role: .destructive) { confirmDelete = true } label: {
                        Label("Delete match", systemImage: "trash")
                    }
                    .buttonStyle(.actionRow(.red))
                }
                .padding(.vertical, 2)
            }
            .scrollIndicators(.hidden)
            .frame(width: 360)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 12)
        .background(Theme.background.ignoresSafeArea())
        .onAppear {
            info = manifest?.info ?? .init()
            if let reelURL { player = AVPlayer(url: reelURL) }
        }
        .onChange(of: info.title) { _ in savedInfo = false }
        .onChange(of: info.location) { _ in savedInfo = false }
        .onChange(of: info.notes) { _ in savedInfo = false }
        .onChange(of: info.players) { _ in savedInfo = false }
        .onDisappear { player?.pause() }
        .confirmationDialog("Delete this match?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete the recording and reel", role: .destructive) {
                model.delete(matchID)
                model.showMatches()
            }
        } message: {
            Text("The video is removed from this iPhone. Copies saved to Photos or YouTube stay.")
        }
    }

    @ViewBuilder private var actions: some View {
        Button {
            Task { await run("Saving the reel…") {
                guard let reelURL else { return }
                try await PhotosSaver.save(videos: [reelURL])
                model.current?.update { $0.savedReelToPhotos = true }
            } }
        } label: { Label(manifest?.savedReelToPhotos == true ? "Reel saved to Photos" : "Save reel to Photos",
                         systemImage: manifest?.savedReelToPhotos == true ? "checkmark.circle.fill" : "photo.on.rectangle") }
            .buttonStyle(.actionRow)
            .disabled(reelURL == nil)
        Button {
            Task { await run("Saving the full recording…") {
                guard let manifest else { return }
                try await PhotosSaver.save(videos: manifest.parts.map { MatchStore.file($0.file, of: matchID) })
                model.current?.update { $0.savedRawToPhotos = true }
            } }
        } label: { Label(manifest?.savedRawToPhotos == true ? "Recording saved to Photos" : "Save full recording to Photos",
                         systemImage: manifest?.savedRawToPhotos == true ? "checkmark.circle.fill" : "film.stack") }
            .buttonStyle(.actionRow)
        if let reelURL {
            ShareLink(item: reelURL) { Label("Share the reel", systemImage: "square.and.arrow.up") }
                .buttonStyle(.actionRow)
        }
        if YouTubeUploader.isConfigured {
            youtube
        } else {
            Label("YouTube upload: not set up in this build", systemImage: "play.rectangle")
                .foregroundStyle(.secondary).font(.footnote).padding(.horizontal, 14).padding(.top, 4)
        }
    }

    @ViewBuilder private var youtube: some View {
        if let videoID = manifest?.upload.videoID, let url = URL(string: "https://youtu.be/\(videoID)") {
            Link(destination: url) { Label("On YouTube", systemImage: "play.rectangle.fill") }
                .buttonStyle(.actionRow)
        } else if manifest?.upload.sessionURL != nil {
            Button {
                Task { await run("Resuming the upload…") {
                    guard let manifest, let session = manifest.upload.sessionURL, let reelURL else { return }
                    try await YouTubeUploader.shared.resume(file: reelURL, sessionURL: session, matchID: matchID)
                } }
            } label: { Label("Uploading to YouTube… (tap to resume)", systemImage: "arrow.clockwise") }
                .buttonStyle(.actionRow)
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
                .buttonStyle(.actionRow)
        }
        Button("Disconnect Google account") { Task { await YouTubeUploader.signOut() } }
            .font(.footnote).padding(.horizontal, 14).padding(.top, 2)
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
