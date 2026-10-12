// FinishView.swift
//
// After Stop: analysis catches up with the camera and the reel is cut --
// seconds when analysis kept up, longer if the phone ran hot. The match info
// is typed here meanwhile, so the wait is not idle.
import SwiftUI

struct FinishView: View {
    @Environment(AppModel.self) private var model
    @State private var info = MatchManifest.Info()

    var body: some View {
        HStack(alignment: .top, spacing: 30) {
            VStack(alignment: .leading, spacing: 14) {
                Text("Match info").font(.title2.bold())
                MatchInfoForm(info: $info)
                Button("Save info") { model.current?.updateInfo(info) }.buttonStyle(.bordered)
            }
            .frame(maxWidth: 420)
            VStack(spacing: 16) {
                progress
            }
            .frame(maxWidth: .infinity)
        }
        .padding(30)
        .onAppear { info = model.current?.manifest.info ?? .init() }
    }

    @ViewBuilder private var progress: some View {
        let controller = model.current
        switch controller?.manifest.state {
        case .finished where controller?.manifest.reel == nil:
            Image(systemName: "questionmark.circle").font(.system(size: 54)).foregroundStyle(.orange)
            Text(controller?.manifest.lastError ?? "No points were found.").font(.title3)
            Text("The full recording is kept; save it to Photos from the match.").foregroundStyle(.secondary)
            Button("Open the match") {
                if let controller { controller.updateInfo(info); model.showDetail(controller.manifest.id) }
            }
            .buttonStyle(.borderedProminent)
        case .finished:
            Image(systemName: "checkmark.circle.fill").font(.system(size: 54)).foregroundStyle(.green)
            Text("The reel is ready").font(.title2.bold())
            if let reel = controller?.manifest.reel {
                Text("\(reel.points) points · \(RecordView.clock(reel.seconds))").foregroundStyle(.secondary)
            }
            Button("Open the reel") {
                if let controller { controller.updateInfo(info); model.showDetail(controller.manifest.id) }
            }
            .buttonStyle(.borderedProminent)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 54)).foregroundStyle(.orange)
            Text(controller?.manifest.lastError ?? "Something went wrong.").multilineTextAlignment(.center)
            HStack {
                Button("Try again") {
                    if let manifest = controller?.manifest { Task { await model.finish(manifest) } }
                }
                .buttonStyle(.borderedProminent)
                Button("Back to matches") { controller?.updateInfo(info); model.home() }
                    .buttonStyle(.bordered)
            }
        default:
            ProgressView().controlSize(.large)
            Text(controller?.finishingStep.isEmpty == false ? controller!.finishingStep : "Finishing…").font(.headline)
            let status = controller?.analysisStatus
            if let status, status.segmentsWaiting > 0 {
                Text("\(status.segmentsWaiting * 2) s of recording left to analyse").foregroundStyle(.secondary)
            }
            Text("Keep the app open until the reel is ready.").font(.footnote).foregroundStyle(.secondary)
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
        .textFieldStyle(.roundedBorder)
    }
}
