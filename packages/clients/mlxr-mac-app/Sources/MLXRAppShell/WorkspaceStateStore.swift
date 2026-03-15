import Foundation
import MLXRAppDomain

public struct WorkspaceStateStore: @unchecked Sendable {
    private let fileManager: FileManager
    private let rootDirectory: URL
    private let manifestURL: URL
    private let legacyManifestURL: URL

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let applicationSupport =
            (try? fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )) ?? fileManager.temporaryDirectory
        let root = applicationSupport
            .appending(path: "MLXR", directoryHint: .isDirectory)
            .appending(path: "CreationState", directoryHint: .isDirectory)
        self.rootDirectory = root
        self.manifestURL = root.appending(path: "app-state.json")
        self.legacyManifestURL = applicationSupport
            .appending(path: "MLXR", directoryHint: .isDirectory)
            .appending(path: "StudioState", directoryHint: .isDirectory)
            .appending(path: "app-state.json")
        try? ensureDirectories()
    }

    public func load() throws -> AppPresentationState {
        let sourceURL: URL
        if fileManager.fileExists(atPath: manifestURL.path()) {
            sourceURL = manifestURL
        } else if fileManager.fileExists(atPath: legacyManifestURL.path()) {
            sourceURL = legacyManifestURL
        } else {
            return AppPresentationState()
        }
        let data = try Data(contentsOf: sourceURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(AppPresentationState.self, from: data)
    }

    public func persist(_ state: AppPresentationState) throws {
        try ensureDirectories()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func ensureDirectories() throws {
        if !fileManager.fileExists(atPath: rootDirectory.path()) {
            try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        }
    }
}
