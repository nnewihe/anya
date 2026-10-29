// PhotosSaver.swift
//
// Save a video into the user's photo library. Asks only for "add" access:
// the app never reads the library.
import Photos

enum PhotosSaver {
    enum SaveError: LocalizedError {
        case notAllowed
        var errorDescription: String? { "Anya isn't allowed to add to Photos. Allow it in Settings > Privacy > Photos." }
    }

    static func save(videos urls: [URL]) async throws {
        var status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if status == .notDetermined { status = await PHPhotoLibrary.requestAuthorization(for: .addOnly) }
        guard status == .authorized || status == .limited else { throw SaveError.notAllowed }
        try await PHPhotoLibrary.shared().performChanges {
            for url in urls { PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url) }
        }
    }
}
