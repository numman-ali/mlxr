import Foundation

public typealias JSONMap = [String: JSONValue]

public enum RecommendationTier: String, Codable, Sendable, CaseIterable {
    case recommended
    case advanced
}

public enum SupportLevel: String, Codable, Sendable, CaseIterable {
    case promoted
    case supported
}

public enum ModelInstallStatus: String, Codable, Sendable {
    case installed
    case alreadyInstalled = "already_installed"
}

public enum ModelInstallOperationPhase: String, Codable, Sendable {
    case queued
    case resolving
    case authRequired = "auth_required"
    case downloading
    case converting
    case registering
    case completed
    case failed
    case cancelled

    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled, .authRequired:
            true
        default:
            false
        }
    }
}

public enum ModelRemoveStatus: String, Codable, Sendable {
    case removed
}

public struct RuntimeHealth: Codable, Sendable, Hashable {
    public var status: String
    public var transportDefault: String

    public init(status: String, transportDefault: String) {
        self.status = status
        self.transportDefault = transportDefault
    }
}

public struct SourceMaterializationRequest: Codable, Sendable, Hashable {
    public var mode: String = "provider-cache-ref"

    public init(mode: String = "provider-cache-ref") {
        self.mode = mode
    }
}

public struct SourceAuth: Codable, Sendable, Hashable {
    public var tokenRef: String?

    public init(tokenRef: String? = nil) {
        self.tokenRef = tokenRef
    }
}

public struct SourcePolicy: Codable, Sendable, Hashable {
    public var allowRemoteCode: Bool

    public init(allowRemoteCode: Bool = false) {
        self.allowRemoteCode = allowRemoteCode
    }
}

public struct SourceRef: Codable, Sendable, Hashable {
    public var provider: String
    public var locator: JSONMap
    public var materialization: SourceMaterializationRequest
    public var auth: SourceAuth
    public var policy: SourcePolicy
    public var familyHint: String?
    public var metadata: JSONMap

    public init(
        provider: String,
        locator: JSONMap,
        materialization: SourceMaterializationRequest = .init(),
        auth: SourceAuth = .init(),
        policy: SourcePolicy = .init(),
        familyHint: String? = nil,
        metadata: JSONMap = [:]
    ) {
        self.provider = provider
        self.locator = locator
        self.materialization = materialization
        self.auth = auth
        self.policy = policy
        self.familyHint = familyHint
        self.metadata = metadata
    }
}

public struct AuthRequirements: Codable, Sendable, Hashable {
    public var required: Bool
    public var supported: [String]
    public var message: String?
}

public struct SourceFileRecord: Codable, Sendable, Hashable, Identifiable {
    public var path: String
    public var sizeBytes: Int?
    public var digest: String?

    public var id: String { path }
}

public struct ResolvedSource: Codable, Sendable, Hashable {
    public var provider: String
    public var locator: JSONMap
    public var pinnedRef: String?
    public var accessState: String
    public var license: String?
    public var remoteCodeRequired: Bool
    public var authRequirements: AuthRequirements
    public var files: [SourceFileRecord]
    public var metadata: JSONMap
}

public struct ProvenanceRecord: Codable, Sendable, Hashable {
    public var provider: String
    public var locator: JSONMap
    public var resolvedRef: String?
    public var license: String?
    public var accessState: String
    public var remoteCodeRequired: Bool
    public var remoteCodeApproved: Bool
    public var blobDigests: [String: String]
    public var fetchedAt: Date
    public var metadata: JSONMap
}

public struct ProviderInspectionResult: Codable, Sendable, Hashable {
    public var bytesTotal: Int?
    public var metadata: JSONMap
}

public struct SourceInspectionTimings: Codable, Sendable, Hashable {
    public var resolveMs: Double
    public var providerInspectMs: Double
    public var familyInspectMs: Double?
    public var provenanceMs: Double
    public var totalMs: Double
}

public struct FamilyInspectionResult: Codable, Sendable, Hashable {
    public var family: String
    public var variant: String?
    public var tasks: [String]
    public var schedulerClass: String?
    public var metadata: JSONMap
}

public struct SourceInspectionResult: Codable, Sendable, Hashable {
    public var resolvedSource: ResolvedSource
    public var providerInspection: ProviderInspectionResult
    public var provenance: ProvenanceRecord
    public var familyInspection: FamilyInspectionResult?
    public var timingsMs: SourceInspectionTimings?
}

public struct SourceRegistrationRecord: Codable, Sendable, Hashable, Identifiable {
    public var sourceId: String
    public var source: SourceRef
    public var resolvedSource: ResolvedSource
    public var provenance: ProvenanceRecord
    public var familyHint: String?
    public var createdAt: Date
    public var updatedAt: Date
    public var metadata: JSONMap

    public var id: String { sourceId }
}

public struct HardwareTier: Codable, Sendable, Hashable {
    public var tier: String
    public var memoryGb: Int?
    public var notes: String?
    public var disabledProfiles: [String]
}

public struct PolicyDescriptor: Codable, Sendable, Hashable {
    public var license: String?
    public var accessState: String
    public var remoteCodeRequired: Bool
    public var remoteCodeApproved: Bool
    public var redistributionState: String?
}

public struct ExtensionSchemaDescriptor: Codable, Sendable, Hashable {
    public var namespace: String
    public var version: String
}

public struct CapabilityDescriptor: Codable, Sendable, Hashable, Identifiable {
    public var modelId: String
    public var artifactDigest: String
    public var family: String
    public var familyVariant: String?
    public var tasks: [String]
    public var modalitiesIn: [String]
    public var modalitiesOut: [String]
    public var constraints: JSONMap
    public var conditioning: JSONMap
    public var profilesByTask: [String: [String]]
    public var streaming: [String: Bool]
    public var artifactsOut: [String]
    public var schedulerClass: String
    public var hardwareTiers: [HardwareTier]
    public var dependencies: JSONMap
    public var policy: PolicyDescriptor
    public var extensionsSchema: ExtensionSchemaDescriptor?
    public var metadata: JSONMap

    public var id: String { modelId }
}

public struct PortableArtifactRecord: Codable, Sendable, Hashable {
    public var modelId: String
    public var artifactDigest: String
    public var family: String
    public var familyVariant: String?
    public var formatVersion: String
    public var weightFormat: String
    public var storageKey: String
    public var capability: CapabilityDescriptor
    public var provenance: ProvenanceRecord
    public var metadata: JSONMap
}

public struct ModelRecord: Codable, Sendable, Hashable, Identifiable {
    public var modelId: String
    public var family: String
    public var source: SourceRef?
    public var artifact: PortableArtifactRecord?
    public var loaded: Bool
    public var capability: CapabilityDescriptor?

    public var id: String { modelId }
}

public struct SupportedModelDescriptor: Codable, Sendable, Hashable, Identifiable {
    public var modelId: String
    public var displayName: String
    public var family: String
    public var familyVariant: String?
    public var recommendationTier: RecommendationTier
    public var supportLevel: SupportLevel
    public var tasks: [String]
    public var provider: String
    public var sourceSummary: String
    public var license: String?
    public var accessState: String
    public var installed: Bool
    public var installable: Bool
    public var notes: String?

    public var id: String { modelId }
}

public struct SupportedModelSourcePreview: Codable, Sendable, Hashable, Identifiable {
    public var role: String
    public var provider: String
    public var locator: JSONMap
    public var resolvedRef: String?
    public var accessState: String
    public var license: String?
    public var bytesTotal: Int?
    public var authRequirements: AuthRequirements
    public var remoteCodeRequired: Bool
    public var remoteCodeApproved: Bool

    public var id: String { role }
}

public struct SupportedModelPreview: Codable, Sendable, Hashable {
    public var supportedModel: SupportedModelDescriptor
    public var sources: [SupportedModelSourcePreview]
    public var totalSourceBytes: Int?
    public var authRequired: Bool
    public var authMessage: String?
}

public struct ModelInstallResult: Codable, Sendable, Hashable {
    public var status: ModelInstallStatus
    public var model: ModelRecord
    public var supportedModel: SupportedModelDescriptor
}

public struct ModelInstallOperationRecord: Codable, Sendable, Hashable, Identifiable {
    public var operationId: String
    public var modelId: String
    public var phase: ModelInstallOperationPhase
    public var supportedModel: SupportedModelDescriptor
    public var preview: SupportedModelPreview?
    public var result: ModelInstallResult?
    public var error: String?
    public var createdAt: Date
    public var updatedAt: Date
    public var startedAt: Date?
    public var finishedAt: Date?
    public var metadata: JSONMap

    public var id: String { operationId }
}

public struct InstalledModelDetails: Codable, Sendable, Hashable {
    public var model: ModelRecord
    public var supportedModel: SupportedModelDescriptor?
    public var managedStorageKey: String?
    public var managedSizeBytes: Int?
    public var referencedSourceIds: [String]
}

public struct ModelRemoveResult: Codable, Sendable, Hashable {
    public var status: ModelRemoveStatus
    public var modelId: String
    public var artifactDigest: String?
    public var removedSourceIds: [String]
    public var removedStorageKey: String?
}

public struct ArtifactConversionRequest: Codable, Sendable, Hashable {
    public var sourceId: String?
    public var sourceBindings: [String: String]?
    public var family: String?
    public var modelId: String
    public var precision: String
    public var targetFormat: String
    public var options: JSONMap

    public init(
        sourceId: String? = nil,
        sourceBindings: [String: String]? = nil,
        family: String? = nil,
        modelId: String,
        precision: String = "bf16",
        targetFormat: String = "mlx_portable_bundle",
        options: JSONMap = [:]
    ) {
        self.sourceId = sourceId
        self.sourceBindings = sourceBindings
        self.family = family
        self.modelId = modelId
        self.precision = precision
        self.targetFormat = targetFormat
        self.options = options
    }
}

public struct ArtifactConversionTimings: Codable, Sendable, Hashable {
    public var fetchMsByRole: [String: Double]
    public var fetchTotalMs: Double
    public var familyConvertMs: Double
    public var persistMs: Double
    public var totalMs: Double
}

public struct ArtifactConversionResult: Codable, Sendable, Hashable {
    public var artifact: PortableArtifactRecord
    public var model: ModelRecord
    public var timingsMs: ArtifactConversionTimings?
}

public extension RuntimeHealth {
    static let offline = RuntimeHealth(status: "offline", transportDefault: "uds")
}
