import Foundation

public enum LibraryAssetOrigin: String, Codable, Sendable, Hashable {
    case generated
    case imported
}

public enum LibraryAssetKind: String, Codable, CaseIterable, Sendable, Hashable {
    case image
    case video
    case audio
    case other

    public static func infer(mediaType: String?, fileExtension: String?) -> LibraryAssetKind {
        if mediaType?.hasPrefix("image/") == true {
            return .image
        }
        if mediaType?.hasPrefix("video/") == true {
            return .video
        }
        if mediaType?.hasPrefix("audio/") == true {
            return .audio
        }

        switch fileExtension?.lowercased() {
        case "png", "jpg", "jpeg", "webp", "heic":
            return .image
        case "mp4", "mov", "m4v":
            return .video
        case "wav", "mp3", "m4a", "aac", "flac":
            return .audio
        default:
            return .other
        }
    }
}

public enum LibraryAssetFilter: String, CaseIterable, Identifiable, Sendable {
    case all = "All"
    case images = "Images"
    case videos = "Videos"
    case audio = "Audio"

    public var id: String { rawValue }

    public func includes(_ asset: LibraryAsset) -> Bool {
        switch self {
        case .all:
            true
        case .images:
            asset.kind == .image
        case .videos:
            asset.kind == .video
        case .audio:
            asset.kind == .audio
        }
    }
}

public struct ImportedAssetRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var title: String
    public var sourcePath: String
    public var storageKey: String
    public var mediaType: String?
    public var kind: LibraryAssetKind
    public var importedAt: Date

    public init(
        id: String,
        title: String,
        sourcePath: String,
        storageKey: String,
        mediaType: String?,
        kind: LibraryAssetKind,
        importedAt: Date
    ) {
        self.id = id
        self.title = title
        self.sourcePath = sourcePath
        self.storageKey = storageKey
        self.mediaType = mediaType
        self.kind = kind
        self.importedAt = importedAt
    }
}

public struct LibraryAsset: Identifiable, Hashable, Sendable {
    public var id: String
    public var origin: LibraryAssetOrigin
    public var kind: LibraryAssetKind
    public var title: String
    public var prompt: String
    public var modelId: String?
    public var task: ProductTask?
    public var createdAt: Date
    public var mediaType: String?
    public var jobId: String?
    public var artifact: OutputArtifactRecord?
    public var job: JobRecord?
    public var importedAsset: ImportedAssetRecord?
    public var runGroupId: String?
    public var collectionIds: [String]
    public var isFavorite: Bool
    public var lastUsedAt: Date?

    public init(
        id: String,
        origin: LibraryAssetOrigin,
        kind: LibraryAssetKind,
        title: String,
        prompt: String,
        modelId: String?,
        task: ProductTask?,
        createdAt: Date,
        mediaType: String?,
        jobId: String?,
        artifact: OutputArtifactRecord?,
        job: JobRecord?,
        importedAsset: ImportedAssetRecord?,
        runGroupId: String? = nil,
        collectionIds: [String] = [],
        isFavorite: Bool = false,
        lastUsedAt: Date? = nil
    ) {
        self.id = id
        self.origin = origin
        self.kind = kind
        self.title = title
        self.prompt = prompt
        self.modelId = modelId
        self.task = task
        self.createdAt = createdAt
        self.mediaType = mediaType
        self.jobId = jobId
        self.artifact = artifact
        self.job = job
        self.importedAsset = importedAsset
        self.runGroupId = runGroupId
        self.collectionIds = collectionIds
        self.isFavorite = isFavorite
        self.lastUsedAt = lastUsedAt
    }

    public var isImported: Bool { origin == .imported }
    public var isGenerated: Bool { origin == .generated }
    public var isImage: Bool { kind == .image }
    public var isVideo: Bool { kind == .video }
    public var isAudio: Bool { kind == .audio }

    public var filename: String? {
        if let artifact, let filename = artifact.filename, !filename.isEmpty {
            return filename
        }
        if let importedAsset {
            return importedAsset.title
        }
        return nil
    }

    public var displayTitle: String {
        switch origin {
        case .generated:
            if let promptHeadline = Self.promptHeadline(from: prompt) {
                return promptHeadline
            }
            return filename ?? title
        case .imported:
            return title
        }
    }

    public var referenceKind: WorkflowReferenceKind? {
        switch kind {
        case .image:
            .image
        case .video:
            .video
        case .audio:
            .audio
        case .other:
            nil
        }
    }

    public var subtitle: String {
        if !prompt.isEmpty {
            return prompt
        }
        if let task {
            return task.title
        }
        if isImported {
            return "Imported asset"
        }
        return "Generated asset"
    }

    public var sourceSummary: String {
        switch origin {
        case .generated:
            modelId ?? "Generated"
        case .imported:
            "Imported"
        }
    }

    public var searchableText: String {
        [
            title,
            displayTitle,
            filename ?? "",
            prompt,
            modelId ?? "",
            task?.rawValue ?? "",
            task?.title ?? "",
            sourceSummary,
        ]
        .joined(separator: " ")
        .lowercased()
    }

    public static func fromGeneratedEntries(
        _ entries: [LibraryEntry],
        metadataById: [String: AssetRecord] = [:]
    ) -> [LibraryAsset] {
        entries.map { entry in
            let metadata = metadataById[entry.id]
            return LibraryAsset(
                id: entry.id,
                origin: .generated,
                kind: LibraryAssetKind.infer(
                    mediaType: entry.artifact.mediaType,
                    fileExtension: entry.artifact.filename.flatMap { URL(filePath: $0).pathExtension }
                ),
                title: entry.title,
                prompt: entry.prompt,
                modelId: entry.job.request.modelId,
                task: entry.task,
                createdAt: entry.job.updatedAt,
                mediaType: entry.artifact.mediaType,
                jobId: entry.job.jobId,
                artifact: entry.artifact,
                job: entry.job,
                importedAsset: nil,
                runGroupId: entry.job.request.context?.runGroupId,
                collectionIds: metadata?.collectionIds ?? [],
                isFavorite: metadata?.isFavorite ?? false,
                lastUsedAt: metadata?.lastUsedAt
            )
        }
    }

    public static func fromImported(
        _ records: [ImportedAssetRecord],
        metadataById: [String: AssetRecord] = [:]
    ) -> [LibraryAsset] {
        records.map { record in
            let metadata = metadataById[record.id]
            return LibraryAsset(
                id: record.id,
                origin: .imported,
                kind: record.kind,
                title: record.title,
                prompt: "",
                modelId: nil,
                task: nil,
                createdAt: record.importedAt,
                mediaType: record.mediaType,
                jobId: nil,
                artifact: nil,
                job: nil,
                importedAsset: record,
                runGroupId: nil,
                collectionIds: metadata?.collectionIds ?? [],
                isFavorite: metadata?.isFavorite ?? false,
                lastUsedAt: metadata?.lastUsedAt
            )
        }
    }

    private static func promptHeadline(from prompt: String) -> String? {
        let normalized = prompt
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            return nil
        }
        if normalized.count <= 72 {
            return normalized
        }
        let cutoff = normalized.index(normalized.startIndex, offsetBy: 69)
        let truncated = String(normalized[..<cutoff]).trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(truncated)…"
    }
}

public struct LibraryAssetGroup: Identifiable, Hashable, Sendable {
    public let id: String
    public let assets: [LibraryAsset]

    public init(id: String, assets: [LibraryAsset]) {
        self.id = id
        self.assets = assets
    }

    public var primaryAsset: LibraryAsset {
        assets[0]
    }
}
