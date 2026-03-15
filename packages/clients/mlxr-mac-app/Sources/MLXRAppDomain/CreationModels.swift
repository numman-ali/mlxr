import Foundation

public struct ComposerSeedRequest: Identifiable, Sendable, Hashable {
    public let id: UUID
    public var workspaceId: String?
    public var task: ProductTask
    public var prompt: String?
    public var focusedAssetId: String?
    public var referenceAssetIds: [String]

    public init(
        id: UUID = UUID(),
        workspaceId: String? = nil,
        task: ProductTask,
        prompt: String? = nil,
        focusedAssetId: String? = nil,
        referenceAssetIds: [String] = []
    ) {
        self.id = id
        self.workspaceId = workspaceId
        self.task = task
        self.prompt = prompt
        self.focusedAssetId = focusedAssetId
        self.referenceAssetIds = referenceAssetIds
    }
}

public enum AssetSortMode: String, CaseIterable, Sendable, Codable {
    case newest = "Newest"
    case lastUsed = "Last Used"
}

public struct CollectionRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var title: String
    public var createdAt: Date

    public init(
        id: String = UUID().uuidString,
        title: String,
        createdAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
    }
}

public struct WorkspaceRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var title: String
    public var coverAssetId: String?
    public var createdAt: Date
    public var lastOpenedAt: Date

    public init(
        id: String = UUID().uuidString,
        title: String,
        coverAssetId: String? = nil,
        createdAt: Date = .now,
        lastOpenedAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.coverAssetId = coverAssetId
        self.createdAt = createdAt
        self.lastOpenedAt = lastOpenedAt
    }
}

public enum RunGroupState: String, Codable, Sendable {
    case queued
    case running
    case completed
    case failed
    case cancelled
}

public struct RunGroupRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var workspaceId: String
    public var collectionId: String?
    public var task: ProductTask
    public var title: String
    public var sourceAssetIds: [String]
    public var jobIds: [String]
    public var assetIds: [String]
    public var variationCount: Int
    public var createdAt: Date
    public var updatedAt: Date
    public var state: RunGroupState

    public init(
        id: String = UUID().uuidString,
        workspaceId: String,
        collectionId: String? = nil,
        task: ProductTask,
        title: String,
        sourceAssetIds: [String] = [],
        jobIds: [String] = [],
        assetIds: [String] = [],
        variationCount: Int = 1,
        createdAt: Date = .now,
        updatedAt: Date = .now,
        state: RunGroupState = .queued
    ) {
        self.id = id
        self.workspaceId = workspaceId
        self.collectionId = collectionId
        self.task = task
        self.title = title
        self.sourceAssetIds = sourceAssetIds
        self.jobIds = jobIds
        self.assetIds = assetIds
        self.variationCount = variationCount
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.state = state
    }
}

public struct AssetRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var isFavorite: Bool
    public var collectionIds: [String]
    public var lastUsedAt: Date?

    public init(
        id: String,
        isFavorite: Bool = false,
        collectionIds: [String] = [],
        lastUsedAt: Date? = nil
    ) {
        self.id = id
        self.isFavorite = isFavorite
        self.collectionIds = collectionIds
        self.lastUsedAt = lastUsedAt
    }
}

public enum PackCategory: String, Codable, CaseIterable, Sendable {
    case looks = "Looks"
    case motion = "Motion Packs"
    case control = "Control Packs"
}

public struct PackRecord: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var title: String
    public var summary: String
    public var category: PackCategory
    public var family: String
    public var tasks: [ProductTask]
    public var familyExtensions: JSONMap
    public var workflowVariant: String?
    public var controlVariant: String?
    public var conditioningAttentionStrength: Double?

    public init(
        id: String,
        title: String,
        summary: String,
        category: PackCategory,
        family: String,
        tasks: [ProductTask],
        familyExtensions: JSONMap = [:],
        workflowVariant: String? = nil,
        controlVariant: String? = nil,
        conditioningAttentionStrength: Double? = nil
    ) {
        self.id = id
        self.title = title
        self.summary = summary
        self.category = category
        self.family = family
        self.tasks = tasks
        self.familyExtensions = familyExtensions
        self.workflowVariant = workflowVariant
        self.controlVariant = controlVariant
        self.conditioningAttentionStrength = conditioningAttentionStrength
    }

    public func supports(task: ProductTask) -> Bool {
        tasks.contains(task)
    }
}

public struct CreationDraft: Codable, Hashable, Sendable {
    public var workspaceId: String
    public var task: ProductTask
    public var prompt: String
    public var negativePrompt: String
    public var selectedModelId: String
    public var selectedPackId: String?
    public var qualityPreset: QualityPreset
    public var aspectPreset: AspectPreset
    public var durationPreset: DurationPreset
    public var variationCount: Int
    public var useCustomSettings: Bool
    public var manualWidth: Double
    public var manualHeight: Double
    public var manualFrames: Double
    public var manualFps: Double
    public var manualSteps: Double
    public var manualGuidance: Double
    public var seed: String
    public var artifactFormat: String
    public var showAdvanced: Bool
    public var selectedAssetId: String?
    public var referenceAssetIds: [String]
    public var lastActiveRunGroupId: String?
    public var preferredDisplayedAssetId: String?

    public init(
        workspaceId: String = "default-workspace",
        task: ProductTask = .imageGenerate,
        prompt: String = "",
        negativePrompt: String = "",
        selectedModelId: String = "",
        selectedPackId: String? = nil,
        qualityPreset: QualityPreset = .standard,
        aspectPreset: AspectPreset = .square,
        durationPreset: DurationPreset = .medium,
        variationCount: Int = 1,
        useCustomSettings: Bool = false,
        manualWidth: Double = 1024,
        manualHeight: Double = 1024,
        manualFrames: Double = 81,
        manualFps: Double = 24,
        manualSteps: Double = 20,
        manualGuidance: Double = 4.0,
        seed: String = "",
        artifactFormat: String = "png",
        showAdvanced: Bool = false,
        selectedAssetId: String? = nil,
        referenceAssetIds: [String] = [],
        lastActiveRunGroupId: String? = nil,
        preferredDisplayedAssetId: String? = nil
    ) {
        self.workspaceId = workspaceId
        self.task = task
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.selectedModelId = selectedModelId
        self.selectedPackId = selectedPackId
        self.qualityPreset = qualityPreset
        self.aspectPreset = aspectPreset
        self.durationPreset = durationPreset
        self.variationCount = variationCount
        self.useCustomSettings = useCustomSettings
        self.manualWidth = manualWidth
        self.manualHeight = manualHeight
        self.manualFrames = manualFrames
        self.manualFps = manualFps
        self.manualSteps = manualSteps
        self.manualGuidance = manualGuidance
        self.seed = seed
        self.artifactFormat = artifactFormat
        self.showAdvanced = showAdvanced
        self.selectedAssetId = selectedAssetId
        self.referenceAssetIds = referenceAssetIds
        self.lastActiveRunGroupId = lastActiveRunGroupId
        self.preferredDisplayedAssetId = preferredDisplayedAssetId
    }

    public var hasMeaningfulState: Bool {
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || selectedAssetId != nil
            || !referenceAssetIds.isEmpty
            || lastActiveRunGroupId != nil
    }
}

public struct AppPresentationState: Codable, Hashable, Sendable {
    public var hasCompletedOnboarding: Bool
    public var activeWorkspaceId: String
    public var selectedLibraryWorkspaceId: String?
    public var workspaces: [WorkspaceRecord]
    public var collections: [CollectionRecord]
    public var runGroups: [RunGroupRecord]
    public var dismissedActivityRunGroupIds: [String]
    public var assets: [AssetRecord]
    public var creationDraft: CreationDraft

    private enum CodingKeys: String, CodingKey {
        case hasCompletedOnboarding
        case hasCompletedModelSetup
        case activeWorkspaceId
        case selectedLibraryWorkspaceId
        case workspaces
        case collections
        case runGroups
        case dismissedActivityRunGroupIds
        case assets
        case creationDraft
        case workspaceDraft
    }

    public init(
        hasCompletedOnboarding: Bool = false,
        activeWorkspaceId: String = "default-workspace",
        selectedLibraryWorkspaceId: String? = nil,
        workspaces: [WorkspaceRecord] = [WorkspaceRecord(id: "default-workspace", title: "New Project")],
        collections: [CollectionRecord] = [],
        runGroups: [RunGroupRecord] = [],
        dismissedActivityRunGroupIds: [String] = [],
        assets: [AssetRecord] = [],
        creationDraft: CreationDraft = .init()
    ) {
        self.hasCompletedOnboarding = hasCompletedOnboarding
        self.activeWorkspaceId = activeWorkspaceId
        self.selectedLibraryWorkspaceId = selectedLibraryWorkspaceId
        self.workspaces = workspaces
        self.collections = collections
        self.runGroups = runGroups
        self.dismissedActivityRunGroupIds = dismissedActivityRunGroupIds
        self.assets = assets
        self.creationDraft = creationDraft
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hasCompletedOnboarding =
            try container.decodeIfPresent(Bool.self, forKey: .hasCompletedOnboarding)
            ?? container.decodeIfPresent(Bool.self, forKey: .hasCompletedModelSetup)
            ?? false
        activeWorkspaceId =
            try container.decodeIfPresent(String.self, forKey: .activeWorkspaceId)
            ?? "default-workspace"
        selectedLibraryWorkspaceId =
            try container.decodeIfPresent(String.self, forKey: .selectedLibraryWorkspaceId)
        workspaces =
            try container.decodeIfPresent([WorkspaceRecord].self, forKey: .workspaces)
            ?? [WorkspaceRecord(id: "default-workspace", title: "New Project")]
        collections =
            try container.decodeIfPresent([CollectionRecord].self, forKey: .collections)
            ?? []
        runGroups =
            try container.decodeIfPresent([RunGroupRecord].self, forKey: .runGroups)
            ?? []
        dismissedActivityRunGroupIds =
            try container.decodeIfPresent(
                [String].self,
                forKey: .dismissedActivityRunGroupIds
            )
            ?? []
        assets = try container.decodeIfPresent([AssetRecord].self, forKey: .assets) ?? []
        creationDraft =
            try container.decodeIfPresent(
                CreationDraft.self,
                forKey: .creationDraft
            )
            ?? container.decodeIfPresent(
                CreationDraft.self,
                forKey: .workspaceDraft
            )
            ?? .init()
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hasCompletedOnboarding, forKey: .hasCompletedOnboarding)
        try container.encode(activeWorkspaceId, forKey: .activeWorkspaceId)
        try container.encodeIfPresent(selectedLibraryWorkspaceId, forKey: .selectedLibraryWorkspaceId)
        try container.encode(workspaces, forKey: .workspaces)
        try container.encode(collections, forKey: .collections)
        try container.encode(runGroups, forKey: .runGroups)
        try container.encode(dismissedActivityRunGroupIds, forKey: .dismissedActivityRunGroupIds)
        try container.encode(assets, forKey: .assets)
        try container.encode(creationDraft, forKey: .creationDraft)
    }
}

public struct CreationResolvedSettings: Equatable, Sendable, Hashable, Codable {
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var numInferenceSteps: Int
    public var guidanceScale: Double

    public init(
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Int,
        numInferenceSteps: Int,
        guidanceScale: Double
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.numInferenceSteps = numInferenceSteps
        self.guidanceScale = guidanceScale
    }
}
