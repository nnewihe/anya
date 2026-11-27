// MatchListView.swift
//
// My matches: start a new one, pick up an interrupted one, open a finished reel.
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            switch model.screen {
            case .home: HomeView()
            case .list: MatchListView()
            case .calibrate: CalibrateView()
            case .record: RecordView()
            case .paused: PausedView()
            case .finish: FinishView()
            case .detail(let id): MatchDetailView(matchID: id)
            }
        }
        .tint(Theme.accent)
        .alert("Anya", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.error ?? "")
        }
    }
}

struct MatchListView: View {
    @EnvironmentObject private var model: AppModel
    @State private var deleting: MatchManifest?

    private var finished: [MatchManifest] { model.matches.filter { $0.state == .finished } }
    private var unfinished: [MatchManifest] {
        model.matches.filter { [.calibrated, .paused, .recording, .failed, .finishing].contains($0.state) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                header
                if !unfinished.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        SectionLabel("Interrupted")
                        ForEach(unfinished) { InterruptedCard(match: $0, onDelete: { deleting = $0 }) }
                    }
                }
                if finished.isEmpty && unfinished.isEmpty {
                    emptyState
                } else if !finished.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        SectionLabel("Reels")
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), spacing: 16)], spacing: 16) {
                            ForEach(finished) { match in
                                Button { model.showDetail(match.id) } label: { MatchCard(match: match) }
                                    .buttonStyle(.plain)
                                    .contextMenu {
                                        Button(role: .destructive) { deleting = match } label: {
                                            Label("Delete match", systemImage: "trash")
                                        }
                                    }
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 16)
        }
        .scrollIndicators(.hidden)
        .background(Theme.background.ignoresSafeArea())
        .onAppear { model.refresh() }
        .confirmationDialog("Delete this match?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible, presenting: deleting) { match in
            Button("Delete the recording and reel", role: .destructive) { model.delete(match.id) }
        } message: { _ in
            Text("The video is removed from this iPhone. Copies saved to Photos or YouTube stay.")
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            CircleIconButton(systemImage: "chevron.left") { model.showHome() }
            Text("My matches").font(Theme.display(30, .heavy))
            Spacer()
            Button { Task { await model.newMatch() } } label: {
                Label("New match", systemImage: "record.circle")
            }
            .buttonStyle(.primary)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "video.badge.plus")
                .font(.system(size: 40, weight: .semibold)).foregroundStyle(Theme.accent)
                .frame(width: 84, height: 84).background(Theme.accent.opacity(0.15), in: Circle())
            Text("No matches yet").font(.title3.bold())
            Text("Put the phone on a tripod behind the baseline, high enough to see the whole court, and tap New match.")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
        }
        .frame(maxWidth: .infinity)
        .card(padding: 32)
    }
}

/// A finished match: its court, title and reel.
private struct MatchCard: View {
    let match: MatchManifest

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .bottomTrailing) {
                MatchThumbnail(matchID: match.id).frame(height: 130)
                if let reel = match.reel {
                    Label(RecordView.clock(reel.seconds), systemImage: "play.fill")
                        .font(.caption.weight(.semibold)).monospacedDigit()
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .glass(Capsule())
                        .padding(10)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(match.displayTitle).font(.headline).lineLimit(1)
                Text(subtitle).font(.footnote).foregroundStyle(.secondary).lineLimit(1)
            }
            .padding(14)
        }
        .background(Theme.card)
        .clipShape(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous).stroke(Theme.stroke))
        .contentShape(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
    }

    private var subtitle: String {
        var parts: [String] = []
        if let reel = match.reel { parts.append(RecordView.points(reel.points)) } else { parts.append("No points found") }
        if !match.info.title.isEmpty { parts.append(match.createdAt.formatted(date: .abbreviated, time: .omitted)) }
        let players = match.info.players.filter { !$0.isEmpty }
        if !players.isEmpty { parts.append(players.joined(separator: " · ")) }
        return parts.joined(separator: " · ")
    }
}

/// A match that stopped before its reel: resume, finish or delete it.
private struct InterruptedCard: View {
    @EnvironmentObject private var model: AppModel
    let match: MatchManifest
    let onDelete: (MatchManifest) -> Void

    var body: some View {
        HStack(spacing: 16) {
            MatchThumbnail(matchID: match.id)
                .frame(width: 120, height: 68)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            VStack(alignment: .leading, spacing: 4) {
                Text(match.displayTitle).font(.headline)
                Text(match.lastError ?? (match.parts.isEmpty ? "Calibrated, not recorded yet."
                                         : "Recording stopped unexpectedly. What's recorded is safe."))
                    .font(.footnote).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer()
            HStack(spacing: 10) {
                if match.state == .paused || match.state == .calibrated {
                    Button(match.parts.isEmpty ? "Record" : "Resume") { Task { await model.resume(match) } }
                        .buttonStyle(.primary(Theme.record))
                }
                if !match.parts.isEmpty {
                    Button("Finish") { Task { await model.finish(match) } }.buttonStyle(.secondary)
                }
                Button { onDelete(match) } label: { Image(systemName: "trash") }
                    .buttonStyle(SecondaryButtonStyle(tint: .red))
            }
        }
        .card(padding: 14, fill: Theme.warning.opacity(0.10))
    }
}
