import AsyncHTTPClient
import Foundation
import MLXRAppDomain
import NIOCore
import NIOFoundationCompat
import NIOHTTP1
import NIOPosix
import UniformTypeIdentifiers

public final class RuntimeClient: RuntimeServing, @unchecked Sendable {
    private let launcher: RuntimeLauncher
    private let httpClient: HTTPClient
    private let udsClient: UnixDomainSocketHTTPClient
    private let cacheDirectory: URL

    public init(launcher: RuntimeLauncher? = nil) throws {
        let eventLoopGroup = MultiThreadedEventLoopGroup.singleton
        self.launcher = try launcher ?? RuntimeLauncher()
        self.httpClient = HTTPClient(eventLoopGroup: eventLoopGroup)
        self.udsClient = UnixDomainSocketHTTPClient(group: eventLoopGroup)
        self.cacheDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library", directoryHint: .isDirectory)
            .appending(path: "Caches", directoryHint: .isDirectory)
            .appending(path: "mlxr-mac-app", directoryHint: .isDirectory)
            .appending(path: "artifacts-cache", directoryHint: .isDirectory)
    }

    deinit {
        try? httpClient.syncShutdown()
    }

    public func status() async throws -> RuntimeStatusSnapshot {
        let configuration = try await healthyConfiguration()
        let health: RuntimeHealth = try await request(.GET, path: "/health", configuration: configuration)
        let launchedByApp = await launcher.didLaunchProcess
        return RuntimeStatusSnapshot(
            health: health,
            runtimeHome: configuration.runtimeHome,
            socketPath: launcher.socketPath,
            logFile: configuration.logFile,
            launchedByApp: launchedByApp
        )
    }

    /// Whether the daemon process launched by the app has terminated unexpectedly.
    public var isRuntimeProcessDead: Bool {
        get async { await launcher.isProcessDead }
    }

    public func listSupportedModels() async throws -> [SupportedModelDescriptor] {
        try await request(.GET, path: "/v1/models/supported")
    }

    public func previewSupportedModel(modelId: String) async throws -> SupportedModelPreview {
        try await request(.GET, path: "/v1/models/supported/\(modelId)/preview")
    }

    public func listModels() async throws -> [ModelRecord] {
        try await request(.GET, path: "/v1/models")
    }

    public func modelDetails(modelId: String) async throws -> InstalledModelDetails {
        try await request(.GET, path: "/v1/models/\(modelId)/details")
    }

    public func listCapabilities() async throws -> [CapabilityDescriptor] {
        try await request(.GET, path: "/v1/capabilities")
    }

    public func listJobs() async throws -> [JobRecord] {
        try await request(.GET, path: "/v1/jobs")
    }

    public func listModelInstalls() async throws -> [ModelInstallOperationRecord] {
        try await request(.GET, path: "/v1/model-installs")
    }

    public func installModel(modelId: String) async throws -> ModelInstallResult {
        let body: JSONMap = ["model_id": .string(modelId)]
        let result: ModelInstallResult = try await request(
            .POST,
            path: "/v1/models/install",
            jsonBody: body
        )
        return result
    }

    public func enqueueModelInstall(modelId: String) async throws -> ModelInstallOperationRecord {
        let body: JSONMap = ["model_id": .string(modelId)]
        return try await request(.POST, path: "/v1/model-installs", jsonBody: body)
    }

    public func cancelModelInstall(operationId: String) async throws -> ModelInstallOperationRecord {
        try await request(.POST, path: "/v1/model-installs/\(operationId)/cancel")
    }

    public func removeModel(modelId: String) async throws -> ModelRemoveResult {
        try await request(.DELETE, path: "/v1/models/\(modelId)")
    }

    public func inspectSource(_ sourceRef: SourceRef) async throws -> SourceInspectionResult {
        try await request(.POST, path: "/v1/sources/inspect", encodableBody: sourceRef)
    }

    public func registerSource(_ sourceRef: SourceRef) async throws -> SourceRegistrationRecord {
        try await request(.POST, path: "/v1/sources/register", encodableBody: sourceRef)
    }

    public func convertArtifact(_ request: ArtifactConversionRequest) async throws -> ArtifactConversionResult {
        try await self.request(.POST, path: "/v1/artifacts/convert", encodableBody: request)
    }

    public func importFile(at fileURL: URL, kind: WorkflowReferenceKind) async throws -> InputHandleRecord {
        let configuration = try await healthyConfiguration()
        let payload = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let filename = fileURL.lastPathComponent
        let mediaType = mediaType(for: fileURL, kind: kind)
        let query = [
            URLQueryItem(name: "filename", value: filename),
            URLQueryItem(name: "media_type", value: mediaType),
            URLQueryItem(name: "role", value: kind.rawValue),
        ]
        return try await request(
            .POST,
            path: "/v1/inputs/import-file",
            configuration: configuration,
            queryItems: query,
            bodyData: payload,
            additionalHeaders: [("content-type", "application/octet-stream")]
        )
    }

    public func plan(intent: WorkflowIntent) async throws -> WorkflowPlanResult {
        try await request(.POST, path: "/v1/workflows/plan", encodableBody: intent)
    }

    public func run(intent: WorkflowIntent) async throws -> WorkflowRunResult {
        let body: JSONMap = [
            "intent": try JSONValueEncoder.encodeToValue(intent),
        ]
        return try await request(.POST, path: "/v1/workflows/run", jsonBody: body)
    }

    public func cancel(jobId: String) async throws -> JobRecord {
        try await request(.POST, path: "/v1/jobs/\(jobId)/cancel")
    }

    public func streamEvents(jobId: String) -> AsyncThrowingStream<RuntimeEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let configuration = try await healthyConfiguration()
                    switch configuration.transport {
                    case let .unixDomainSocket(pathURL):
                        try await udsClient.streamEvents(
                            socketPath: pathURL.path,
                            uri: requestURI(for: "/v1/jobs/\(jobId)/events", queryItems: []),
                            headers: requestHeaders(
                                configuration: configuration,
                                defaultAccept: nil,
                                additionalHeaders: [("accept", "text/event-stream")]
                            ),
                            decoder: Self.decoder,
                            continuation: continuation
                        )
                    case .loopbackHTTP:
                        let request = try buildRequest(
                            method: .GET,
                            path: "/v1/jobs/\(jobId)/events",
                            configuration: configuration,
                            queryItems: [],
                            additionalHeaders: [("accept", "text/event-stream")],
                            defaultAccept: nil
                        )
                        let delegate = EventStreamDelegate(
                            decoder: Self.decoder,
                            errorBuilder: Self.error,
                            continuation: continuation
                        )
                        let task = httpClient.execute(
                            request: request,
                            delegate: delegate,
                            deadline: .now() + .seconds(300)
                        )
                        try await task.get()
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func exportOutput(
        artifactId: String,
        destinationPath: URL,
        overwrite: Bool
    ) async throws -> ArtifactExportResult {
        let body: JSONMap = [
            "destination_path": .string(destinationPath.path),
            "overwrite": .bool(overwrite),
        ]
        let result: ArtifactExportResult = try await request(
            .POST,
            path: "/v1/outputs/\(artifactId)/export",
            jsonBody: body
        )
        return result
    }

    public func cachedDownloadURL(for artifact: OutputArtifactRecord) async throws -> URL {
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let filename = artifact.filename ?? "\(artifact.artifactId).\(artifact.artifactFormat)"
        let localDirectory = cacheDirectory.appending(path: artifact.jobId, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: localDirectory, withIntermediateDirectories: true)
        let localURL = localDirectory.appending(path: filename)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        let configuration = try await healthyConfiguration()
        let response = try await rawRequest(
            .GET,
            path: "/v1/outputs/\(artifact.artifactId)/download",
            configuration: configuration
        )
        guard response.status.code < 400 else {
            let data = try Self.collectBody(from: response)
            throw Self.error(from: Int(response.status.code), payload: data)
        }
        let payload = try Self.collectBody(from: response)
        do {
            try payload.write(to: localURL, options: Data.WritingOptions.atomic)
        } catch {
            throw RuntimeBridgeError.outputDownloadFailed(error.localizedDescription)
        }
        return localURL
    }

    private func healthyConfiguration() async throws -> RuntimeConnectionConfiguration {
        _ = try await launcher.ensureRunning { [weak self] configuration in
            guard let self else { return false }
            do {
                let response = try await self.rawRequest(.GET, path: "/health", configuration: configuration)
                return response.status == .ok
            } catch {
                return false
            }
        }
        return launcher.configuration
    }

    private func request<Response: Decodable>(
        _ method: HTTPMethod,
        path: String,
        encodableBody: some Encodable
    ) async throws -> Response {
        let configuration = try await healthyConfiguration()
        return try await request(method, path: path, configuration: configuration, encodableBody: encodableBody)
    }

    private func request<Response: Decodable>(
        _ method: HTTPMethod,
        path: String,
        jsonBody: JSONMap? = nil
    ) async throws -> Response {
        let configuration = try await healthyConfiguration()
        return try await request(method, path: path, configuration: configuration, jsonBody: jsonBody)
    }

    private func request<Response: Decodable>(
        _ method: HTTPMethod,
        path: String,
        configuration: RuntimeConnectionConfiguration,
        encodableBody: some Encodable
    ) async throws -> Response {
        let data = try Self.encoder.encode(encodableBody)
        return try await request(
            method,
            path: path,
            configuration: configuration,
            bodyData: data,
            additionalHeaders: [("content-type", "application/json")]
        )
    }

    private func request<Response: Decodable>(
        _ method: HTTPMethod,
        path: String,
        configuration: RuntimeConnectionConfiguration,
        jsonBody: JSONMap? = nil
    ) async throws -> Response {
        let bodyData = try jsonBody.map(JSONValueEncoder.encode)
        return try await request(
            method,
            path: path,
            configuration: configuration,
            bodyData: bodyData,
            additionalHeaders: bodyData == nil ? [] : [("content-type", "application/json")]
        )
    }

    private func request<Response: Decodable>(
        _ method: HTTPMethod,
        path: String,
        configuration: RuntimeConnectionConfiguration,
        queryItems: [URLQueryItem] = [],
        bodyData: Data? = nil,
        additionalHeaders: [(String, String)] = []
    ) async throws -> Response {
        let response = try await rawRequest(
            method,
            path: path,
            configuration: configuration,
            queryItems: queryItems,
            bodyData: bodyData,
            additionalHeaders: additionalHeaders
        )
        let payload = try Self.collectBody(from: response)
        guard response.status.code < 400 else {
            throw Self.error(from: Int(response.status.code), payload: payload)
        }
        do {
            return try Self.decoder.decode(Response.self, from: payload)
        } catch {
            throw RuntimeBridgeError.invalidResponse(error.localizedDescription)
        }
    }

    private func rawRequest(
        _ method: HTTPMethod,
        path: String,
        configuration: RuntimeConnectionConfiguration,
        queryItems: [URLQueryItem] = [],
        bodyData: Data? = nil,
        additionalHeaders: [(String, String)] = []
    ) async throws -> HTTPClient.Response {
        switch configuration.transport {
        case let .unixDomainSocket(pathURL):
            let response = try await udsClient.execute(
                method: method,
                socketPath: pathURL.path,
                uri: requestURI(for: path, queryItems: queryItems),
                headers: requestHeaders(
                    configuration: configuration,
                    defaultAccept: "application/json",
                    additionalHeaders: additionalHeaders
                ),
                body: bodyData
            )
            return HTTPClient.Response(
                host: "localhost",
                status: response.status,
                version: HTTPVersion(major: 1, minor: 1),
                headers: response.headers,
                body: ByteBuffer(bytes: response.body)
            )
        case .loopbackHTTP:
            let request = try buildRequest(
                method: method,
                path: path,
                configuration: configuration,
                queryItems: queryItems,
                additionalHeaders: additionalHeaders,
                bodyData: bodyData
            )
            return try await httpClient.execute(
                request: request,
                deadline: .now() + .seconds(300)
            ).get()
        }
    }

    private func buildRequest(
        method: HTTPMethod,
        path: String,
        configuration: RuntimeConnectionConfiguration,
        queryItems: [URLQueryItem],
        additionalHeaders: [(String, String)],
        defaultAccept: String? = "application/json",
        bodyData: Data? = nil
    ) throws -> HTTPClient.Request {
        let url = try requestURL(for: path, configuration: configuration, queryItems: queryItems)
        let headers = requestHeaders(
            configuration: configuration,
            defaultAccept: defaultAccept,
            additionalHeaders: additionalHeaders
        )
        return try HTTPClient.Request(
            url: url,
            method: method,
            headers: headers,
            body: bodyData.map(HTTPClient.Body.bytes)
        )
    }

    private func requestURL(
        for path: String,
        configuration: RuntimeConnectionConfiguration,
        queryItems: [URLQueryItem]
    ) throws -> URL {
        switch configuration.transport {
        case let .loopbackHTTP(baseURL, _):
            let url = baseURL.appending(path: requestURI(for: path, queryItems: []))
            return append(queryItems: queryItems, to: url)
        case .unixDomainSocket:
            throw RuntimeBridgeError.invalidResponse("Loopback URL construction was requested for a Unix-domain socket transport.")
        }
    }

    private func requestURI(for path: String, queryItems: [URLQueryItem]) -> String {
        let requestPath = path.hasPrefix("/") ? path : "/\(path)"
        guard !queryItems.isEmpty else {
            return requestPath
        }
        var components = URLComponents()
        components.path = requestPath
        components.queryItems = queryItems
        return components.string ?? requestPath
    }

    private func requestHeaders(
        configuration: RuntimeConnectionConfiguration,
        defaultAccept: String?,
        additionalHeaders: [(String, String)]
    ) -> HTTPHeaders {
        var headers = HTTPHeaders()
        if let defaultAccept {
            headers.add(name: "accept", value: defaultAccept)
        }

        switch configuration.transport {
        case let .loopbackHTTP(_, token):
            headers.add(name: "authorization", value: "Bearer \(token)")
        case .unixDomainSocket:
            break
        }

        for (name, value) in additionalHeaders {
            headers.replaceOrAdd(name: name, value: value)
        }
        return headers
    }

    private func append(queryItems: [URLQueryItem], to url: URL) -> URL {
        guard !queryItems.isEmpty else {
            return url
        }
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        components?.queryItems = queryItems
        return components?.url ?? url
    }

    private static func collectBody(from response: HTTPClient.Response) throws -> Data {
        guard var body = response.body else {
            return Data()
        }
        return body.readData(length: body.readableBytes) ?? Data()
    }

    fileprivate static func error(from statusCode: Int, payload: Data) -> RuntimeBridgeError {
        if let detail = Self.errorDetail(from: payload) {
            return .api(statusCode: statusCode, detail: detail)
        }
        return .api(statusCode: statusCode, detail: String(decoding: payload, as: UTF8.self))
    }

    private func mediaType(for fileURL: URL, kind: WorkflowReferenceKind) -> String {
        if let type = UTType(filenameExtension: fileURL.pathExtension),
           let mimeType = type.preferredMIMEType {
            return mimeType
        }
        switch kind {
        case .image:
            return "image/\(fileURL.pathExtension.lowercased())"
        case .video:
            return "video/\(fileURL.pathExtension.lowercased())"
        case .audio:
            return "audio/\(fileURL.pathExtension.lowercased())"
        case .lora:
            return "application/octet-stream"
        }
    }

    private static func errorDetail(from payload: Data) -> String? {
        guard let object = try? decoder.decode([String: JSONValue].self, from: payload) else {
            return nil
        }
        if let detail = object["detail"]?.stringValue {
            return detail
        }
        if let issues = object["detail"]?.arrayValue {
            let messages = issues.compactMap { issue -> String? in
                guard let issueObject = issue.objectValue else {
                    return issue.stringValue
                }
                let location = issueObject["loc"]?.arrayValue?
                    .compactMap { $0.stringValue ?? $0.integerValue.map(String.init) }
                    .joined(separator: " > ")
                let message = issueObject["msg"]?.stringValue ?? issueObject["message"]?.stringValue
                switch (location, message) {
                case let (location?, message?):
                    return "\(location): \(message)"
                case let (_, message?):
                    return message
                default:
                    return nil
                }
            }
            if !messages.isEmpty {
                return messages.joined(separator: "\n")
            }
        }
        return nil
    }

    fileprivate static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    fileprivate static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
}
