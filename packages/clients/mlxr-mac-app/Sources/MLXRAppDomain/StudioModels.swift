import Foundation

public struct StudioOpenRequest: Identifiable, Sendable, Hashable {
    public let id: UUID
    public var task: ProductTask
    public var prompt: String?
    public var focusedAssetId: String?
    public var referenceAssetIds: [String]

    public init(
        id: UUID = UUID(),
        task: ProductTask,
        prompt: String? = nil,
        focusedAssetId: String? = nil,
        referenceAssetIds: [String] = []
    ) {
        self.id = id
        self.task = task
        self.prompt = prompt
        self.focusedAssetId = focusedAssetId
        self.referenceAssetIds = referenceAssetIds
    }
}

public enum StudioWorkflowAvailability: Sendable, Hashable {
    case available(ProductTask)
    case comingSoon
}

public struct StudioWorkflowOption: Identifiable, Sendable, Hashable {
    public let id: String
    public let title: String
    public let subtitle: String
    public let icon: String
    public let availability: StudioWorkflowAvailability

    public init(
        id: String,
        title: String,
        subtitle: String,
        icon: String,
        availability: StudioWorkflowAvailability
    ) {
        self.id = id
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.availability = availability
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
    public var createdAt: Date
    public var lastOpenedAt: Date

    public init(
        id: String = UUID().uuidString,
        title: String,
        createdAt: Date = .now,
        lastOpenedAt: Date = .now
    ) {
        self.id = id
        self.title = title
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

public struct StudioWorkspaceDraft: Codable, Hashable, Sendable {
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
    public var hasCompletedModelSetup: Bool
    public var activeWorkspaceId: String
    public var workspaces: [WorkspaceRecord]
    public var collections: [CollectionRecord]
    public var runGroups: [RunGroupRecord]
    public var assets: [AssetRecord]
    public var workspaceDraft: StudioWorkspaceDraft

    public init(
        hasCompletedModelSetup: Bool = false,
        activeWorkspaceId: String = "default-workspace",
        workspaces: [WorkspaceRecord] = [WorkspaceRecord(id: "default-workspace", title: "Current Workspace")],
        collections: [CollectionRecord] = [],
        runGroups: [RunGroupRecord] = [],
        assets: [AssetRecord] = [],
        workspaceDraft: StudioWorkspaceDraft = .init()
    ) {
        self.hasCompletedModelSetup = hasCompletedModelSetup
        self.activeWorkspaceId = activeWorkspaceId
        self.workspaces = workspaces
        self.collections = collections
        self.runGroups = runGroups
        self.assets = assets
        self.workspaceDraft = workspaceDraft
    }
}

public struct StudioResolvedSettings: Equatable, Sendable, Hashable, Codable {
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
