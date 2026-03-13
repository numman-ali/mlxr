import Foundation

public enum PromptHelperMode: String, CaseIterable, Identifiable, Sendable {
    case off
    case suggest
    case auto

    public var id: String { rawValue }
}

public struct MediaReferenceInput: Identifiable, Hashable, Sendable {
    public var id = UUID()
    public var fileURL: URL
    public var kind: WorkflowReferenceKind
    public var role: String?
    public var metadata: JSONMap

    public init(
        fileURL: URL,
        kind: WorkflowReferenceKind,
        role: String? = "reference",
        metadata: JSONMap = [:]
    ) {
        self.fileURL = fileURL
        self.kind = kind
        self.role = role
        self.metadata = metadata
    }
}

public struct ImageGenerationRequest: Sendable, Hashable {
    public var task: ProductTask
    public var modelId: String
    public var prompt: String
    public var negativePrompt: String
    public var width: Int
    public var height: Int
    public var numInferenceSteps: Int
    public var guidanceScale: Double
    public var seed: Int?
    public var artifactFormat: String
    public var promptHelperMode: PromptHelperMode
    public var quality: WorkflowQuality
    public var references: [MediaReferenceInput]
    public var familyExtensions: JSONMap

    public init(
        task: ProductTask,
        modelId: String,
        prompt: String,
        negativePrompt: String = "",
        width: Int = 1024,
        height: Int = 1024,
        numInferenceSteps: Int = 4,
        guidanceScale: Double = 4.0,
        seed: Int? = nil,
        artifactFormat: String = "png",
        promptHelperMode: PromptHelperMode = .suggest,
        quality: WorkflowQuality = .balanced,
        references: [MediaReferenceInput] = [],
        familyExtensions: JSONMap = [:]
    ) {
        self.task = task
        self.modelId = modelId
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numInferenceSteps = numInferenceSteps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.artifactFormat = artifactFormat
        self.promptHelperMode = promptHelperMode
        self.quality = quality
        self.references = references
        self.familyExtensions = familyExtensions
    }
}

public struct VideoGenerationRequest: Sendable, Hashable {
    public var task: ProductTask
    public var modelId: String
    public var prompt: String
    public var negativePrompt: String
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var numInferenceSteps: Int
    public var guidanceScale: Double
    public var seed: Int?
    public var artifactFormat: String
    public var quality: WorkflowQuality
    public var references: [MediaReferenceInput]
    public var workflowVariant: String?
    public var controlVariant: String?
    public var conditioningAttentionStrength: Double?
    public var windowStartSeconds: Double?
    public var windowEndSeconds: Double?
    public var regenerateVideo: Bool
    public var regenerateAudio: Bool
    public var familyExtensions: JSONMap

    public init(
        task: ProductTask,
        modelId: String,
        prompt: String,
        negativePrompt: String = "",
        width: Int = 704,
        height: Int = 704,
        numFrames: Int = 81,
        fps: Int = 24,
        numInferenceSteps: Int = 40,
        guidanceScale: Double = 3.0,
        seed: Int? = nil,
        artifactFormat: String = "mp4",
        quality: WorkflowQuality = .balanced,
        references: [MediaReferenceInput] = [],
        workflowVariant: String? = nil,
        controlVariant: String? = nil,
        conditioningAttentionStrength: Double? = nil,
        windowStartSeconds: Double? = nil,
        windowEndSeconds: Double? = nil,
        regenerateVideo: Bool = true,
        regenerateAudio: Bool = true,
        familyExtensions: JSONMap = [:]
    ) {
        self.task = task
        self.modelId = modelId
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.numInferenceSteps = numInferenceSteps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.artifactFormat = artifactFormat
        self.quality = quality
        self.references = references
        self.workflowVariant = workflowVariant
        self.controlVariant = controlVariant
        self.conditioningAttentionStrength = conditioningAttentionStrength
        self.windowStartSeconds = windowStartSeconds
        self.windowEndSeconds = windowEndSeconds
        self.regenerateVideo = regenerateVideo
        self.regenerateAudio = regenerateAudio
        self.familyExtensions = familyExtensions
    }
}

public enum AdvancedImportMode: String, CaseIterable, Identifiable, Sendable {
    case huggingFaceSingleRepo
    case trustedLocalBundle
    case ltxMultiSource

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .huggingFaceSingleRepo:
            "Hugging Face Repo"
        case .trustedLocalBundle:
            "Local Bundle"
        case .ltxMultiSource:
            "LTX Multi-Source"
        }
    }
}

public struct AdvancedImportDraft: Sendable, Hashable {
    public var mode: AdvancedImportMode
    public var modelId: String
    public var familyHint: String?
    public var huggingFaceRepo: String
    public var localPath: String
    public var ltxCheckpointRepo: String
    public var ltxUpsamplerRepo: String
    public var ltxTextEncoderRepo: String
    public var ltxDistilledLoRARepo: String

    public init(
        mode: AdvancedImportMode = .huggingFaceSingleRepo,
        modelId: String = "",
        familyHint: String? = nil,
        huggingFaceRepo: String = "",
        localPath: String = "",
        ltxCheckpointRepo: String = "",
        ltxUpsamplerRepo: String = "",
        ltxTextEncoderRepo: String = "",
        ltxDistilledLoRARepo: String = ""
    ) {
        self.mode = mode
        self.modelId = modelId
        self.familyHint = familyHint
        self.huggingFaceRepo = huggingFaceRepo
        self.localPath = localPath
        self.ltxCheckpointRepo = ltxCheckpointRepo
        self.ltxUpsamplerRepo = ltxUpsamplerRepo
        self.ltxTextEncoderRepo = ltxTextEncoderRepo
        self.ltxDistilledLoRARepo = ltxDistilledLoRARepo
    }
}
