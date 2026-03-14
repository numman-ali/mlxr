import Foundation
import MLXRAppDomain
import UniformTypeIdentifiers

public struct ImportedAssetStore {
    private let fileManager: FileManager
    private let baseDirectory: URL
    private let manifestURL: URL

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
            .appending(path: "ImportedAssets", directoryHint: .isDirectory)
        self.baseDirectory = root
        self.manifestURL = root.appending(path: "imported-assets.json")

        try? ensureDirectories()
    }

    func load() throws -> [ImportedAssetRecord] {
        guard fileManager.fileExists(atPath: manifestURL.path()) else {
            return []
        }
        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([ImportedAssetRecord].self, from: data)
            .sorted { $0.importedAt > $1.importedAt }
    }

    func importFiles(at urls: [URL], existing: [ImportedAssetRecord]) throws -> [ImportedAssetRecord] {
        guard !urls.isEmpty else { return existing }
        var records = existing

        for url in urls {
            if records.contains(where: { $0.sourcePath == url.path() }) {
                continue
            }
            let fileName = uniqueFilename(for: url)
            let destination = baseDirectory.appending(path: fileName)
            if fileManager.fileExists(atPath: destination.path()) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: url, to: destination)

            let record = ImportedAssetRecord(
                id: UUID().uuidString,
                title: url.lastPathComponent,
                sourcePath: url.path(),
                storageKey: fileName,
                mediaType: mediaType(for: url),
                kind: LibraryAssetKind.infer(
                    mediaType: mediaType(for: url),
                    fileExtension: url.pathExtension
                ),
                importedAt: .now
            )
            records.insert(record, at: 0)
        }

        try persist(records)
        return records.sorted { $0.importedAt > $1.importedAt }
    }

    func remove(recordId: String, existing: [ImportedAssetRecord]) throws -> [ImportedAssetRecord] {
        var records = existing
        guard let record = records.first(where: { $0.id == recordId }) else {
            return records
        }

        let fileURL = storedFileURL(for: record)
        if fileManager.fileExists(atPath: fileURL.path()) {
            try fileManager.removeItem(at: fileURL)
        }
        records.removeAll { $0.id == recordId }
        try persist(records)
        return records.sorted { $0.importedAt > $1.importedAt }
    }

    func fileURL(for record: ImportedAssetRecord) -> URL {
        storedFileURL(for: record)
    }

    private func ensureDirectories() throws {
        if !fileManager.fileExists(atPath: baseDirectory.path()) {
            try fileManager.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        }
    }

    private func persist(_ records: [ImportedAssetRecord]) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(records)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func uniqueFilename(for url: URL) -> String {
        let stem = UUID().uuidString.lowercased()
        let ext = url.pathExtension
        guard !ext.isEmpty else { return stem }
        return "\(stem).\(ext)"
    }

    private func mediaType(for url: URL) -> String? {
        guard let type = UTType(filenameExtension: url.pathExtension.lowercased()) else {
            return nil
        }
        if let preferred = type.preferredMIMEType {
            return preferred
        }
        if type.conforms(to: .image) { return "image/\(url.pathExtension.lowercased())" }
        if type.conforms(to: .movie) { return "video/\(url.pathExtension.lowercased())" }
        if type.conforms(to: .audio) { return "audio/\(url.pathExtension.lowercased())" }
        return nil
    }

    private func storedFileURL(for record: ImportedAssetRecord) -> URL {
        baseDirectory.appending(path: record.storageKey)
    }
}
