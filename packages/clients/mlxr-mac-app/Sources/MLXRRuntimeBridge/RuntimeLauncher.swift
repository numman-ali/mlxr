import Foundation
import MLXRAppDomain

public actor RuntimeLauncher {
    public static let healthPollIntervalNanoseconds: UInt64 = 500_000_000

    public nonisolated let configuration: RuntimeConnectionConfiguration

    private let repoRoot: URL
    private let pythonExecutable: URL
    private var launchedProcess: Process?
    private var processTerminated = false

    public init() throws {
        let resolvedRepoRoot = try Self.resolveRepoRoot()
        let resolvedRuntimeHome = Self.resolveRuntimeHome()
        let resolvedSocket = Self.resolveSocketPath(runtimeHome: resolvedRuntimeHome)
        self.configuration = RuntimeConnectionConfiguration(
            transport: .unixDomainSocket(path: resolvedSocket),
            runtimeHome: resolvedRuntimeHome,
            logFile: resolvedRuntimeHome
                .appending(path: "logs", directoryHint: .isDirectory)
                .appending(path: "control-plane-stdio.log")
        )
        self.repoRoot = resolvedRepoRoot
        self.pythonExecutable = try Self.resolvePythonExecutable(repoRoot: resolvedRepoRoot)
    }

    public static func resolveRepoRoot() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXR_MAC_APP_REPO_ROOT"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        var candidate = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            candidate.deleteLastPathComponent()
        }
        let readme = candidate.appending(path: "README.md")
        guard FileManager.default.fileExists(atPath: readme.path) else {
            throw RuntimeBridgeError.runtimeRootNotFound
        }
        return candidate
    }

    public static func resolveRuntimeHome() -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXR_MAC_APP_RUNTIME_HOME"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if let shared = ProcessInfo.processInfo.environment["MLX_RUNTIME_HOME"] {
            return URL(fileURLWithPath: shared, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".mlx-runtime", directoryHint: .isDirectory)
    }

    public static func resolveSocketPath(runtimeHome: URL) -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXR_MAC_APP_UDS_PATH"] {
            return URL(fileURLWithPath: override)
        }
        let candidate = runtimeHome
            .appending(path: "temp", directoryHint: .isDirectory)
            .appending(path: "control-plane.sock")
        if candidate.path.utf8.count <= 100 {
            return candidate
        }
        return URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "mlxr-control-plane.sock")
    }

    public static func resolvePythonExecutable(repoRoot: URL) throws -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXR_MAC_APP_PYTHON"] {
            return URL(fileURLWithPath: override)
        }
        let venvPython = repoRoot
            .appending(path: ".venv", directoryHint: .isDirectory)
            .appending(path: "bin", directoryHint: .isDirectory)
            .appending(path: "python")
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return venvPython
        }
        if let virtualEnv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"] {
            let virtualEnvPython = URL(fileURLWithPath: virtualEnv)
                .appending(path: "bin", directoryHint: .isDirectory)
                .appending(path: "python")
            if FileManager.default.isExecutableFile(atPath: virtualEnvPython.path) {
                return virtualEnvPython
            }
        }
        throw RuntimeBridgeError.pythonExecutableNotFound(venvPython)
    }

    /// Whether the app launched the daemon itself (vs. discovering an already-running one).
    public var didLaunchProcess: Bool {
        launchedProcess != nil
    }

    /// Whether the launched process has terminated (crash or clean exit).
    public var isProcessDead: Bool {
        processTerminated
    }

    public func ensureRunning(
        healthCheck: @escaping @Sendable (RuntimeConnectionConfiguration) async -> Bool
    ) async throws -> RuntimeStatusSnapshot {
        // Check if there is already a healthy daemon (external or app-launched).
        if await healthCheck(configuration) {
            return RuntimeStatusSnapshot(
                health: RuntimeHealth(status: "ok", transportDefault: "uds"),
                runtimeHome: configuration.runtimeHome,
                socketPath: socketPath,
                logFile: configuration.logFile,
                launchedByApp: launchedProcess != nil
            )
        }

        // If we previously launched a process and it has died, clear it so we can relaunch.
        if let launchedProcess, !launchedProcess.isRunning {
            self.launchedProcess = nil
            self.processTerminated = false
        }

        try startRuntime()

        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if await healthCheck(configuration) {
                return RuntimeStatusSnapshot(
                    health: RuntimeHealth(status: "ok", transportDefault: "uds"),
                    runtimeHome: configuration.runtimeHome,
                    socketPath: socketPath,
                    logFile: configuration.logFile,
                    launchedByApp: true
                )
            }
            try await Task.sleep(nanoseconds: Self.healthPollIntervalNanoseconds)
        }

        throw RuntimeBridgeError.failedToStartRuntime(logFile: configuration.logFile)
    }

    public nonisolated var socketPath: URL? {
        switch configuration.transport {
        case let .unixDomainSocket(path):
            path
        case .loopbackHTTP:
            nil
        }
    }

    private func startRuntime() throws {
        // Don't spawn a duplicate if we already have a running process.
        if launchedProcess?.isRunning == true {
            return
        }

        let fileManager = FileManager.default
        let logDirectory = configuration.logFile.deletingLastPathComponent()
        let socketDirectory = socketPath?.deletingLastPathComponent()
        try fileManager.createDirectory(at: configuration.runtimeHome, withIntermediateDirectories: true, attributes: nil)
        try fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true, attributes: nil)
        if let socketDirectory {
            try fileManager.createDirectory(at: socketDirectory, withIntermediateDirectories: true, attributes: nil)
        }
        if let socketPath, fileManager.fileExists(atPath: socketPath.path) {
            try? fileManager.removeItem(at: socketPath)
        }
        if !fileManager.fileExists(atPath: configuration.logFile.path) {
            _ = fileManager.createFile(atPath: configuration.logFile.path, contents: nil)
        }

        let process = Process()
        process.executableURL = pythonExecutable
        process.arguments = ["-m", "mlxr.core.server"]
        process.currentDirectoryURL = repoRoot

        var environment = ProcessInfo.processInfo.environment
        environment["MLX_RUNTIME_HOME"] = configuration.runtimeHome.path
        if let socketPath {
            environment["MLX_RUNTIME_UDS_PATH"] = socketPath.path
        }
        environment.removeValue(forKey: "MLX_RUNTIME_HTTP_HOST")
        environment.removeValue(forKey: "MLX_RUNTIME_HTTP_PORT")
        environment.removeValue(forKey: "MLX_RUNTIME_HTTP_TOKEN")
        environment.removeValue(forKey: "MLX_RUNTIME_ALLOWED_ORIGINS")
        process.environment = environment

        let logHandle = try FileHandle(forWritingTo: configuration.logFile)
        try logHandle.seekToEnd()
        process.standardOutput = logHandle
        process.standardError = logHandle

        // Detect daemon termination so the app can surface it.
        process.terminationHandler = { [weak self] _ in
            guard let self else { return }
            Task { await self.markProcessTerminated() }
        }

        try process.run()
        launchedProcess = process
        processTerminated = false
    }

    private func markProcessTerminated() {
        processTerminated = true
    }
}
