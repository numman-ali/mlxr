import Foundation

public enum TaskCategory: String, CaseIterable, Identifiable, Sendable, Codable {
    case image
    case video

    public var id: String { rawValue }
}

public enum ProductTask: String, CaseIterable, Identifiable, Sendable, Codable {
    case imageGenerate = "image.generate"
    case imageEdit = "image.edit"
    case videoGenerate = "video.generate"
    case videoConditionImage = "video.condition.image"
    case videoConditionAudio = "video.condition.audio"
    case videoConditionVideo = "video.condition.video"
    case videoInterpolate = "video.interpolate"
    case videoRetake = "video.retake"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .imageGenerate:
            "Make Image"
        case .imageEdit:
            "Edit Image"
        case .videoGenerate:
            "Make Video"
        case .videoConditionImage:
            "Animate Image"
        case .videoConditionAudio:
            "Video From Sound"
        case .videoConditionVideo:
            "Guide With Video"
        case .videoInterpolate:
            "Blend Frames"
        case .videoRetake:
            "Retake Clip"
        }
    }

    public var subtitle: String {
        switch self {
        case .imageGenerate:
            "Create a still image from a prompt"
        case .imageEdit:
            "Change or refine an existing image"
        case .videoGenerate:
            "Create a clip from text alone"
        case .videoConditionImage:
            "Turn a still into motion"
        case .videoConditionAudio:
            "Drive a clip from sound"
        case .videoConditionVideo:
            "Steer motion with a source clip"
        case .videoInterpolate:
            "Create motion between key frames"
        case .videoRetake:
            "Redo part of a clip without starting over"
        }
    }

    public var category: TaskCategory {
        switch self {
        case .imageGenerate, .imageEdit:
            .image
        case .videoGenerate, .videoConditionImage, .videoConditionAudio, .videoConditionVideo, .videoInterpolate, .videoRetake:
            .video
        }
    }

    public var isAdvancedByDefault: Bool {
        switch self {
        case .imageGenerate, .imageEdit, .videoGenerate, .videoConditionImage, .videoConditionAudio:
            false
        case .videoConditionVideo, .videoInterpolate, .videoRetake:
            true
        }
    }

    public static func from(rawTask: String) -> ProductTask? {
        ProductTask(rawValue: rawTask)
    }
}

public struct ModelCatalogItem: Identifiable, Hashable, Sendable {
    public var modelId: String
    public var displayName: String
    public var family: String
    public var familyVariant: String?
    public var recommendationTier: RecommendationTier
    public var supportLevel: SupportLevel
    public var installed: Bool
    public var installable: Bool
    public var loaded: Bool
    public var tasks: [ProductTask]
    public var capability: CapabilityDescriptor?
    public var sourceSummary: String?
    public var notes: String?
    public var license: String?
    public var accessState: String

    public var id: String { modelId }

    public var isRecommended: Bool { recommendationTier == .recommended }

    public var isPromoted: Bool { supportLevel == .promoted }

    public var sectionLabel: String {
        if !installed && installable {
            return isRecommended ? "Recommended to install" : "Advanced install"
        }
        return isRecommended ? "Recommended" : "Advanced"
    }

    public var statusLabel: String {
        if loaded {
            return "Loaded"
        }
        if installed {
            return "Installed"
        }
        return installable ? "Available" : "Unavailable"
    }

    public func supports(_ task: ProductTask) -> Bool {
        tasks.contains(task)
    }

    public var implementedPipelineVariants: [String] {
        capability?.metadata.object("implemented_surface")?.array("pipeline_variants")?.compactMap(\.stringValue) ?? []
    }
}

public struct CatalogSnapshot: Sendable {
    public var items: [ModelCatalogItem]

    public init(
        supportedModels: [SupportedModelDescriptor],
        installedModels: [ModelRecord],
        capabilities: [CapabilityDescriptor]
    ) {
        let capabilityByModel = Dictionary(uniqueKeysWithValues: capabilities.map { ($0.modelId, $0) })
        let installedByModel = Dictionary(uniqueKeysWithValues: installedModels.map { ($0.modelId, $0) })
        let supportedByModel = Dictionary(uniqueKeysWithValues: supportedModels.map { ($0.modelId, $0) })

        var merged: [ModelCatalogItem] = supportedModels.map { supported in
            let installed = installedByModel[supported.modelId]
            let capability = capabilityByModel[supported.modelId] ?? installed?.capability
            return ModelCatalogItem(
                modelId: supported.modelId,
                displayName: supported.displayName,
                family: supported.family,
                familyVariant: capability?.familyVariant ?? supported.familyVariant,
                recommendationTier: supported.recommendationTier,
                supportLevel: supported.supportLevel,
                installed: supported.installed || installed != nil,
                installable: supported.installable,
                loaded: installed?.loaded ?? false,
                tasks: Self.tasks(for: supported.tasks, capability: capability),
                capability: capability,
                sourceSummary: supported.sourceSummary,
                notes: supported.notes,
                license: supported.license ?? capability?.policy.license,
                accessState: supported.accessState
            )
        }

        let installedOnly = installedModels
            .filter { supportedByModel[$0.modelId] == nil }
            .map { model in
                let capability = capabilityByModel[model.modelId] ?? model.capability
                return ModelCatalogItem(
                    modelId: model.modelId,
                    displayName: Self.derivedDisplayName(for: model, capability: capability),
                    family: model.family,
                    familyVariant: capability?.familyVariant ?? model.artifact?.familyVariant,
                    recommendationTier: .advanced,
                    supportLevel: .supported,
                    installed: true,
                    installable: false,
                    loaded: model.loaded,
                    tasks: Self.tasks(for: capability?.tasks ?? [], capability: capability),
                    capability: capability,
                    sourceSummary: model.source?.provider,
                    notes: "Installed outside the curated catalog.",
                    license: capability?.policy.license ?? model.artifact?.provenance.license,
                    accessState: capability?.policy.accessState ?? model.artifact?.provenance.accessState ?? "unknown"
                )
            }

        merged.append(contentsOf: installedOnly)
        self.items = merged.sorted {
            if $0.recommendationTier != $1.recommendationTier {
                return $0.recommendationTier == .recommended
            }
            if $0.installed != $1.installed {
                return $0.installed && !$1.installed
            }
            return $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
    }

    public var installedItems: [ModelCatalogItem] {
        items.filter(\.installed)
    }

    public var recommendedItems: [ModelCatalogItem] {
        items.filter(\.isRecommended)
    }

    public var recommendedInstalledItems: [ModelCatalogItem] {
        recommendedItems.filter(\.installed)
    }

    public var recommendedAvailableItems: [ModelCatalogItem] {
        recommendedItems.filter { !$0.installed && $0.installable }
    }

    public func items(for category: TaskCategory) -> [ModelCatalogItem] {
        items.filter { item in item.tasks.contains(where: { $0.category == category }) }
    }

    public func items(for task: ProductTask) -> [ModelCatalogItem] {
        items.filter { $0.supports(task) }
    }

    public func recommendedItems(for task: ProductTask) -> [ModelCatalogItem] {
        items(for: task).filter(\.isRecommended)
    }

    public func advancedItems(for task: ProductTask) -> [ModelCatalogItem] {
        items(for: task).filter { !$0.isRecommended }
    }

    public func defaultModel(for task: ProductTask) -> ModelCatalogItem? {
        recommendedItems(for: task).first(where: \.installed)
        ?? recommendedItems(for: task).first
        ?? items(for: task).first(where: \.installed)
        ?? items(for: task).first
    }

    private static func tasks(
        for rawTasks: [String],
        capability: CapabilityDescriptor?
    ) -> [ProductTask] {
        let source = capability?.tasks ?? rawTasks
        return source.compactMap(ProductTask.from(rawTask:))
    }

    private static func derivedDisplayName(
        for model: ModelRecord,
        capability: CapabilityDescriptor?
    ) -> String {
        let variant = capability?.familyVariant ?? model.artifact?.familyVariant
        if let variant, !variant.isEmpty {
            return variant.replacingOccurrences(of: "-", with: " ").capitalized
        }
        return model.modelId.replacingOccurrences(of: "-", with: " ").capitalized
    }
}

public struct LibraryEntry: Identifiable, Hashable, Sendable {
    public var artifact: OutputArtifactRecord
    public var job: JobRecord

    public init(artifact: OutputArtifactRecord, job: JobRecord) {
        self.artifact = artifact
        self.job = job
    }

    public var id: String { artifact.id }

    public var title: String {
        artifact.filename ?? artifact.artifactId
    }

    public var prompt: String {
        job.request.inputs.string("prompt") ?? ""
    }

    public var task: ProductTask? {
        ProductTask.from(rawTask: job.request.task)
    }

    public var family: String {
        job.request.modelId
    }

    public static func flatten(jobs: [JobRecord]) -> [LibraryEntry] {
        jobs
            .sorted { $0.updatedAt > $1.updatedAt }
            .flatMap { job in
                job.artifacts.map { artifact in
                    LibraryEntry(artifact: artifact, job: job)
                }
            }
    }
}
