// MatchListView.swift
//
// Home: start a new match, pick up an interrupted one, open a finished reel.
import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        Group {
            switch model.screen {
            case .list: MatchListView()
            case .calibrate: CalibrateView()
            case .record: RecordView()
            case .paused: PausedView()
            case .finish: FinishView()
            case .detail(let id): MatchDetailView(matchID: id)
            }
        }
        .alert("Anya", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.error ?? "")
        }
    }
}

struct MatchListView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        Task { await model.newMatch() }
                    } label: {
                        Label("New match", systemImage: "video.badge.plus").font(.title3.bold())
                    }
                } footer: {
                    Text("Put the phone on a tripod behind the baseline, high enough to see the whole court.")
                }
                let unfinished = model.matches.filter { [.paused, .recording, .failed, .finishing].contains($0.state) }
                if !unfinished.isEmpty {
                    Section("Interrupted") {
                        ForEach(unfinished) { match in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(match.displayTitle).font(.headline)
                                Text(match.lastError ?? "Recording stopped unexpectedly. What's recorded is safe.")
                                    .font(.footnote).foregroundStyle(.secondary)
                                HStack {
                                    if match.state == .paused {
                                        Button("Resume recording") { Task { await model.resume(match) } }
                                    }
                                    Button("Finish") { Task { await model.finish(match) } }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }
                Section("Matches") {
                    ForEach(model.matches.filter { $0.state == .finished }) { match in
                        Button { model.showDetail(match.id) } label: {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(match.displayTitle).font(.headline)
                                    if let reel = match.reel {
                                        Text("\(reel.points) points · reel \(RecordView.clock(reel.seconds))")
                                            .font(.footnote).foregroundStyle(.secondary)
                                    }
                                }
                                Spacer()
                                Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                            }
                        }
                        .foregroundStyle(.primary)
                    }
                    .onDelete { offsets in
                        let finished = model.matches.filter { $0.state == .finished }
                        for i in offsets { model.delete(finished[i].id) }
                    }
                }
            }
            .navigationTitle("Anya Tennis")
            .onAppear { model.refresh() }
        }
    }
}
