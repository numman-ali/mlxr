import Foundation

public enum OutputDestinationMode: String, Codable, Sendable {
    case runtimeManaged = "runtime_managed"
    case trustedLocalExport = "trusted_local_export"
}

public enum WorkflowQuality: String, Codable, Sendable, CaseIterable {
    case auto
    case fast
    case balanced
    case high
}

public enum WorkflowReferenceKind: String, Codable, Sendable, CaseIterable, Identifiable {
    case image
    case video
    case audio
    case lora

    public var id: String { rawValue }
}

public enum JobState: String, Codable, Sendable, CaseIterable {
    case accepted
    case preparing
    case loadingModel = "loading_model"
    case running
    case streamingOutput = "streaming_output"
    case finalizing
    case completed
    case failed
    case cancelled

    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled:
            true
        default:
            false
        }
    }
}

public enum RuntimeEventKind: String, Codable, Sendable {
    case jobAccepted = "job.accepted"
    case jobPhaseChanged = "job.phase_changed"
    case jobProgress = "job.progress"
    case jobOutputDelta = "job.output.delta"
    case jobSegment = "job.segment"
    case jobAudioChunk = "job.audio_chunk"
    case jobArtifactReady = "job.artifact.ready"
    case jobMetrics = "job.metrics"
    case jobWarning = "job.warning"
    case jobFailed = "job.failed"
    case jobCompleted = "job.completed"
    case jobCancelled = "job.cancelled"
}

public struct WorkflowReference: Codable, Sendable, Hashable, Identifiable {
    public var inputHandle: String?
    public var kind: WorkflowReferenceKind
    public var role: String?
    public var metadata: JSONMap

    public var id: String {
        inputHandle ?? "\(kind.rawValue)-\(role ?? "reference")-\(metadata.hashValue)"
    }

    public init(
        inputHandle: String? = nil,
        kind: WorkflowReferenceKind,
        role: String? = "reference",
        metadata: JSONMap = [:]
    ) {
        self.inputHandle = inputHandle
        self.kind = kind
        self.role = role
        self.metadata = metadata
    }
}

public struct WorkflowPreferences: Codable, Sendable, Hashable {
    public var quality: WorkflowQuality

    public init(quality: WorkflowQuality = .auto) {
        self.quality = quality
    }
}

public struct WorkflowContextMetadata: Codable, Sendable, Hashable {
    public var workspaceId: String?
    public var collectionId: String?
    public var runGroupId: String?
    public var sourceAssetIds: [String]
    public var intentLabel: String?
    public var presetId: String?

    public init(
        workspaceId: String? = nil,
        collectionId: String? = nil,
        runGroupId: String? = nil,
        sourceAssetIds: [String] = [],
        intentLabel: String? = nil,
        presetId: String? = nil
    ) {
        self.workspaceId = workspaceId
        self.collectionId = collectionId
        self.runGroupId = runGroupId
        self.sourceAssetIds = sourceAssetIds
        self.intentLabel = intentLabel
        self.presetId = presetId
    }
}

public struct JobOutputPolicy: Codable, Sendable, Hashable {
    public var artifactFormat: String?
    public var destinationMode: OutputDestinationMode
    public var exportRef: String?
    public var stream: Bool

    public init(
        artifactFormat: String? = nil,
        destinationMode: OutputDestinationMode = .runtimeManaged,
        exportRef: String? = nil,
        stream: Bool = false
    ) {
        self.artifactFormat = artifactFormat
        self.destinationMode = destinationMode
        self.exportRef = exportRef
        self.stream = stream
    }
}

public struct WorkflowIntent: Codable, Sendable, Hashable {
    public var modelId: String
    public var prompt: String
    public var task: String?
    public var negativePrompt: String?
    public var references: [WorkflowReference]
    public var params: JSONMap
    public var output: JobOutputPolicy
    public var preferences: WorkflowPreferences
    public var context: WorkflowContextMetadata?
    public var extensions: JSONMap

    public init(
        modelId: String,
        prompt: String,
        task: String? = nil,
        negativePrompt: String? = nil,
        references: [WorkflowReference] = [],
        params: JSONMap = [:],
        output: JobOutputPolicy = .init(),
        preferences: WorkflowPreferences = .init(),
        context: WorkflowContextMetadata? = nil,
        extensions: JSONMap = [:]
    ) {
        self.modelId = modelId
        self.prompt = prompt
        self.task = task
        self.negativePrompt = negativePrompt
        self.references = references
        self.params = params
        self.output = output
        self.preferences = preferences
        self.context = context
        self.extensions = extensions
    }
}

public struct WorkflowPlan: Codable, Sendable, Hashable {
    public var modelId: String
    public var family: String
    public var schedulerClass: String?
    public var selectedTask: String
    public var selectedProfile: String?
    public var pipelineVariant: String?
    public var resolvedPrompt: String
    public var references: [WorkflowReference]
    public var stages: [WorkflowStageSpec]
    public var warnings: [String]
    public var metadata: JSONMap
}

public struct WorkflowReferenceRequirement: Codable, Sendable, Hashable, Identifiable {
    public var kind: WorkflowReferenceKind
    public var minimumCount: Int
    public var maximumCount: Int?
    public var acceptedRoles: [String]
    public var description: String

    public var id: String {
        let maximumLabel = maximumCount.map(String.init) ?? "many"
        return "\(kind.rawValue)-\(minimumCount)-\(maximumLabel)-\(description)"
    }
}

public struct WorkflowPresentationControlOption: Codable, Sendable, Hashable, Identifiable {
    public var value: String
    public var label: String
    public var `default`: Bool

    public var id: String { value }
}

public struct WorkflowPresentationControls: Codable, Sendable, Hashable {
    public var qualityPresets: [WorkflowPresentationControlOption]
    public var aspectPresets: [WorkflowPresentationControlOption]
    public var durationPresets: [WorkflowPresentationControlOption]
    public var variationCounts: [Int]

    public init(
        qualityPresets: [WorkflowPresentationControlOption] = [],
        aspectPresets: [WorkflowPresentationControlOption] = [],
        durationPresets: [WorkflowPresentationControlOption] = [],
        variationCounts: [Int] = []
    ) {
        self.qualityPresets = qualityPresets
        self.aspectPresets = aspectPresets
        self.durationPresets = durationPresets
        self.variationCounts = variationCounts
    }
}

public struct WorkflowPresentationSubworkflow: Codable, Sendable, Hashable, Identifiable {
    public var task: String
    public var label: String
    public var mode: String
    public var `default`: Bool

    public var id: String { task }
}

public struct WorkflowPresentationReferenceSlot: Codable, Sendable, Hashable, Identifiable {
    public var slotId: String
    public var label: String
    public var kind: WorkflowReferenceKind
    public var description: String?
    public var required: Bool
    public var minimumCount: Int
    public var maximumCount: Int?
    public var acceptedRoles: [String]
    public var allowsMultiple: Bool

    public var id: String { slotId }
}

public struct WorkflowPlanPresentation: Codable, Sendable, Hashable {
    public var primaryMode: String
    public var selectedTask: String
    public var subworkflows: [WorkflowPresentationSubworkflow]
    public var referenceSlots: [WorkflowPresentationReferenceSlot]
    public var controls: WorkflowPresentationControls

    public init(
        primaryMode: String,
        selectedTask: String,
        subworkflows: [WorkflowPresentationSubworkflow] = [],
        referenceSlots: [WorkflowPresentationReferenceSlot] = [],
        controls: WorkflowPresentationControls = .init()
    ) {
        self.primaryMode = primaryMode
        self.selectedTask = selectedTask
        self.subworkflows = subworkflows
        self.referenceSlots = referenceSlots
        self.controls = controls
    }
}

public struct WorkflowPlanReadiness: Codable, Sendable, Hashable {
    public var ready: Bool
    public var blockingIssues: [String]
    public var warnings: [String]
    public var referenceRequirements: [WorkflowReferenceRequirement]
    public var allowedOutputFormats: [String]

    public init(
        ready: Bool = true,
        blockingIssues: [String] = [],
        warnings: [String] = [],
        referenceRequirements: [WorkflowReferenceRequirement] = [],
        allowedOutputFormats: [String] = []
    ) {
        self.ready = ready
        self.blockingIssues = blockingIssues
        self.warnings = warnings
        self.referenceRequirements = referenceRequirements
        self.allowedOutputFormats = allowedOutputFormats
    }
}

public struct WorkflowStageSpec: Codable, Sendable, Hashable, Identifiable {
    public var stageId: String
    public var stageType: String
    public var summary: String?
    public var task: String?
    public var inputs: JSONMap
    public var params: JSONMap
    public var optional: Bool
    public var metadata: JSONMap

    public var id: String { stageId }
}

public struct WorkflowPlanResult: Codable, Sendable, Hashable {
    public var capability: CapabilityDescriptor
    public var plan: WorkflowPlan
    public var readiness: WorkflowPlanReadiness
    public var presentation: WorkflowPlanPresentation = .init(
        primaryMode: "image",
        selectedTask: "image.generate"
    )
}

public struct JobRequest: Codable, Sendable, Hashable {
    public var modelId: String
    public var task: String
    public var inputs: JSONMap
    public var params: JSONMap
    public var output: JobOutputPolicy
    public var context: WorkflowContextMetadata?
    public var extensions: JSONMap
}

public struct OutputArtifactRecord: Codable, Sendable, Hashable, Identifiable {
    public var artifactId: String
    public var artifactFormat: String
    public var role: String
    public var exportable: Bool
    public var metadata: JSONMap
    public var jobId: String
    public var filename: String?
    public var mediaType: String?
    public var sizeBytes: Int?
    public var storageKey: String
    public var createdAt: Date

    public var id: String { artifactId }
}

public struct JobRecord: Codable, Sendable, Hashable, Identifiable {
    public var jobId: String
    public var request: JobRequest
    public var state: JobState
    public var createdAt: Date
    public var updatedAt: Date
    public var error: String?
    public var artifacts: [OutputArtifactRecord]

    public var id: String { jobId }
}

public struct WorkflowRunSubmitResult: Codable, Sendable, Hashable {
    public var jobId: String
    public var record: JobRecord
}

public struct WorkflowRunResult: Codable, Sendable, Hashable {
    public var plan: WorkflowPlan
    public var submit: WorkflowRunSubmitResult
}

public struct InputHandleRecord: Codable, Sendable, Hashable, Identifiable {
    public var handleId: String
    public var mediaType: String?
    public var role: String?
    public var metadata: JSONMap
    public var filename: String?
    public var sizeBytes: Int?
    public var storageKey: String
    public var createdAt: Date

    public var id: String { handleId }
}

public struct RuntimeEvent: Codable, Sendable, Hashable, Identifiable {
    public var jobId: String
    public var kind: RuntimeEventKind
    public var timestamp: Date
    public var phase: String?
    public var data: JSONMap

    public var id: String { "\(jobId)-\(kind.rawValue)-\(timestamp.timeIntervalSince1970)" }
}

public struct ArtifactExportResult: Codable, Sendable, Hashable {
    public var artifactId: String
    public var destinationPath: String
    public var sizeBytes: Int?
}
