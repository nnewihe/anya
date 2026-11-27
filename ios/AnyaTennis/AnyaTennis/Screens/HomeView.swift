// HomeView.swift
//
// The first screen: the app's tagline and the two ways in, "Start a match"
// and "My matches".
import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var model: AppModel

    private var finished: [MatchManifest] { model.matches.filter { $0.state == .finished } }
    private var unfinished: [MatchManifest] {
        model.matches.filter { [.calibrated, .paused, .recording, .failed, .finishing].contains($0.state) }
    }

    var body: some View {
        GeometryReader { geo in
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    topBar
                    Spacer(minLength: 0)
                    hero
                    if !unfinished.isEmpty { interruptedBanner }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 16)
                .frame(minHeight: geo.size.height)
            }
        }
        .scrollIndicators(.hidden)
        .background(Theme.background.ignoresSafeArea())
        .onAppear { model.refresh() }
    }

    // ── top ───────────────────────────────────────────────────────────────

    private var topBar: some View {
        HStack {
            Wordmark()
            Spacer()
            Button { model.showMatches() } label: {
                Label(finished.isEmpty ? "My matches" : "My matches · \(finished.count)", systemImage: "film.stack")
            }
            .buttonStyle(.secondary)
        }
    }

    private var hero: some View {
        HStack(alignment: .center, spacing: 28) {
            VStack(alignment: .leading, spacing: 16) {
                Text("CUT THE DEAD TIME. SEE THE ACTION.")
                    .font(.caption.weight(.bold)).tracking(1.6).foregroundColor(Theme.accent)
                (Text("Watch your match in minutes, ") + Text("not hours.").foregroundColor(Theme.accent))
                    .font(Theme.display(38, .heavy))
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 12) {
                    Button { Task { await model.newMatch() } } label: {
                        Label("Start a match", systemImage: "record.circle")
                    }
                    .buttonStyle(.primary(Theme.accent, large: true))
                    Button { model.showMatches() } label: { Text("My matches") }
                        .buttonStyle(.secondary)
                }
                .padding(.top, 4)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            HeroPoster()
                .frame(width: 180, height: 270)   // the poster's 2:3
        }
    }

    private var interruptedBanner: some View {
        Button { model.showMatches() } label: {
            HStack(spacing: 14) {
                Image(systemName: "pause.circle.fill").font(.title2).foregroundStyle(Theme.warning)
                VStack(alignment: .leading, spacing: 2) {
                    Text(unfinished.count == 1 ? "A match was interrupted" : "\(unfinished.count) matches were interrupted")
                        .font(.headline)
                    Text("What was recorded is safe. Resume or finish it in My matches.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
            }
            .card(padding: 16, fill: Theme.warning.opacity(0.12))
        }
        .buttonStyle(.plain)
    }
}

// ── pieces ────────────────────────────────────────────────────────────────

/// The anyatennis.com poster (www/anya-poster.png).
private struct HeroPoster: View {
    var body: some View {
        Image("Poster").resizable().scaledToFill()
            .clipShape(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous).stroke(Theme.stroke))
            .accessibilityHidden(true)
    }
}
