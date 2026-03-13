import Foundation
import MLXRAppDomain

public enum RuntimeTransport: Sendable, Hashable {
    case unixDomainSocket(path: URL)
    case loopbackHTTP(baseURL: URL, token: String)
}

public struct RuntimeConnectionConfiguration: Sendable, Hashable {
    public var transport: RuntimeTransport
    public var runtimeHome: URL
    public var logFile: URL

    public init(transport: RuntimeTransport, runtimeHome: URL, logFile: URL) {
        self.transport = transport
        self.runtimeHome = runtimeHome
        self.logFile = logFile
    }
}

public struct RuntimeStatusSnapshot: Sendable, Hashable {
    public var health: RuntimeHealth
    public var runtimeHome: URL
    public var socketPath: URL?
    public var logFile: URL
    public var launchedByApp: Bool

    public init(
        health: RuntimeHealth,
        runtimeHome: URL,
        socketPath: URL?,
        logFile: URL,
        launchedByApp: Bool
    ) {
        self.health = health
        self.runtimeHome = runtimeHome
        self.socketPath = socketPath
        self.logFile = logFile
        self.launchedByApp = launchedByApp
    }
}

public enum RuntimeBridgeError: LocalizedError, Sendable {
    case runtimeRootNotFound
    case pythonExecutableNotFound(URL)
    case failedToStartRuntime(logFile: URL)
    case runtimeUnavailable(String)
    case api(statusCode: Int, detail: String)
    case invalidResponse(String)
    case outputDownloadFailed(String)
    case featureUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .runtimeRootNotFound:
            "The MLXR repo root could not be resolved from the app. Set MLXR_MAC_APP_REPO_ROOT to the repo checkout."
        case let .pythonExecutableNotFound(url):
            "The Python runtime for MLXR was not found at \(url.path)."
        case let .failedToStartRuntime(logFile):
            "The MLXR runtime failed to start. Check \(logFile.path)."
        case let .runtimeUnavailable(message):
            message
        case let .api(statusCode, detail):
            "Runtime request failed with HTTP \(statusCode): \(detail)"
        case let .invalidResponse(message):
            "The runtime returned an unexpected response: \(message)"
        case let .outputDownloadFailed(message):
            "Could not materialize the output artifact: \(message)"
        case let .featureUnavailable(message):
            message
        }
    }
}

public protocol RuntimeServing: Sendable {
    func status() async throws -> RuntimeStatusSnapshot
    func listSupportedModels() async throws -> [SupportedModelDescriptor]
    func previewSupportedModel(modelId: String) async throws -> SupportedModelPreview
    func listModels() async throws -> [ModelRecord]
    func modelDetails(modelId: String) async throws -> InstalledModelDetails
    func listCapabilities() async throws -> [CapabilityDescriptor]
    func listJobs() async throws -> [JobRecord]
    func listModelInstalls() async throws -> [ModelInstallOperationRecord]
    func installModel(modelId: String) async throws -> ModelInstallResult
    func enqueueModelInstall(modelId: String) async throws -> ModelInstallOperationRecord
    func cancelModelInstall(operationId: String) async throws -> ModelInstallOperationRecord
    func removeModel(modelId: String) async throws -> ModelRemoveResult
    func inspectSource(_ sourceRef: SourceRef) async throws -> SourceInspectionResult
    func registerSource(_ sourceRef: SourceRef) async throws -> SourceRegistrationRecord
    func convertArtifact(_ request: ArtifactConversionRequest) async throws -> ArtifactConversionResult
    func importFile(at fileURL: URL, kind: WorkflowReferenceKind) async throws -> InputHandleRecord
    func plan(intent: WorkflowIntent) async throws -> WorkflowPlanResult
    func run(intent: WorkflowIntent) async throws -> WorkflowRunResult
    func cancel(jobId: String) async throws -> JobRecord
    func streamEvents(jobId: String) -> AsyncThrowingStream<RuntimeEvent, Error>
    func exportOutput(artifactId: String, destinationPath: URL, overwrite: Bool) async throws -> ArtifactExportResult
    func cachedDownloadURL(for artifact: OutputArtifactRecord) async throws -> URL
}
