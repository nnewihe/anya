// FinishView.swift
//
// After Stop: analysis catches up with the camera and the reel is cut --
// seconds when analysis kept up, longer if the phone ran hot. The match info
// is typed here meanwhile, so the wait is not idle.
import SwiftUI

struct FinishView: View {
    @EnvironmentObject private var model: AppModel
    @State private var info = MatchManifest.Info()

    var body: some View {
        HStack(alignment: .center, spacing: 24) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Match info").font(Theme.display(24, .heavy))
                    Text("Fill this in while the reel is cut.").font(.subheadline).foregroundStyle(.secondary)
                    MatchInfoForm(info: $info)
                    Button { model.current?.updateInfo(info) } label: { Label("Save info", systemImage: "square.and.arrow.down") }
                        .buttonStyle(.secondary)
                }
                .card(padding: 18)
            }
            .scrollIndicators(.hidden)
            .frame(maxWidth: 440)
            VStack(spacing: 16) {
                progress
            }
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
        .background(Theme.background.ignoresSafeArea())
        .onAppear { info = model.current?.manifest.info ?? .init() }
    }

    /// A big status symbol in a tinted circle.
    private func badge(_ systemImage: String, _ tint: Color) -> some View {
        Image(systemName: systemImage)
            .font(.system(size: 40, weight: .bold)).foregroundStyle(tint)
            .frame(width: 92, height: 92)
            .background(tint.opacity(0.16), in: Circle())
    }

    private func openMatch() {
        guard let controller = model.current else { return }
        controller.updateInfo(info)
        model.showDetail(controller.manifest.id)
    }

    @ViewBuilder private var progress: some View {
        let controller = model.current
        switch controller?.manifest.state {
        case .finished where controller?.manifest.reel == nil:
            badge("questionmark", Theme.warning)
            Text(controller?.manifest.lastError ?? "No points were found.").font(.title3.weight(.semibold))
            Text("The full recording is kept; save it to Photos from the match.").foregroundStyle(.secondary)
            Button("Open the match") { openMatch() }.buttonStyle(.primary(Theme.accent, large: true))
        case .finished:
            badge("checkmark", Theme.done)
            Text("The reel is ready").font(Theme.display(28, .heavy))
            if let reel = controller?.manifest.reel {
                Text("\(RecordView.points(reel.points)) · \(RecordView.clock(reel.seconds))")
                    .font(.headline).foregroundStyle(.secondary).monospacedDigit()
            }
            Button { openMatch() } label: { Label("Open the reel", systemImage: "play.fill") }
                .buttonStyle(.primary(Theme.accent, large: true))
        case .failed:
            badge("exclamationmark.triangle", Theme.warning)
            Text(controller?.manifest.lastError ?? "Something went wrong.").font(.headline)
            HStack {
                Button("Try again") {
                    if let manifest = controller?.manifest { Task { await model.finish(manifest) } }
                }
                .buttonStyle(.primary)
                Button("Back to matches") { controller?.updateInfo(info); model.showMatches() }
                    .buttonStyle(.secondary)
            }
        default:
            ProgressView().controlSize(.large).tint(Theme.accent)
                .frame(width: 92, height: 92)
                .background(Theme.accent.opacity(0.12), in: Circle())
            Text(controller?.finishingStep.isEmpty == false ? controller!.finishingStep : "Finishing…")
                .font(Theme.display(24, .bold))
            let status = controller?.analysisStatus
            if let status, status.segmentsWaiting > 0 {
                Text("\(status.segmentsWaiting * 2) s of recording left to analyse")
                    .font(.headline).foregroundStyle(.secondary).monospacedDigit()
            }
            Text("Keep the app open until the reel is ready.").font(.footnote).foregroundStyle(.tertiary)
        }
    }
}

struct MatchInfoForm: View {
    @Binding var info: MatchManifest.Info

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField("Title (e.g. Club final)", text: $info.title)
            HStack {
                TextField("Near player", text: $info.players[0])
                TextField("Partner (doubles)", text: $info.players[1])
            }
            HStack {
                TextField("Far player", text: $info.players[2])
                TextField("Partner (doubles)", text: $info.players[3])
            }
            TextField("Where", text: $info.location)
            TextField("Notes", text: $info.notes, axis: .vertical).lineLimit(2...4)
        }
        .textFieldStyle(FilledFieldStyle())
    }
}

/// A filled rounded field, for dark cards.
struct FilledFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Theme.field, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
