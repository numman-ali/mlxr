import Foundation
import MLXRAppDomain
import MLXRRuntimeBridge
import Observation

@MainActor
@Observable
public final class MLXRAppModel {
    // -- State --
    public var runtimeStatus: RuntimeStatusSnapshot?
    public var catalog = CatalogSnapshot(supportedModels: [], installedModels: [], capabilities: [])
    public var jobs: [JobRecord] = []
    public var installOperations: [ModelInstallOperationRecord] = []
    public var modelPreviews: [String: SupportedModelPreview] = [:]
    public var installedModelDetails: [String: InstalledModelDetails] = [:]
    public var isRefreshing = false
    public var isSubmittingImage = false
    public var isSubmittingVideo = false
    public var promptHelperMode: PromptHelperMode = .suggest
    public var hasCompletedOnboarding = false

    // Per-surface inline errors (shown where the user is working).
    public var imageError: String?
    public var videoError: String?
    public var modelsError: String?
    public var settingsError: String?
    public var globalError: String?
    public var bootstrapError: String?

    // Runtime health flag: true when a daemon we launched has died.
    public var runtimeProcessDied = false

    // Active job progress tracking (jobId -> latest phase).
    public var activeJobPhases: [String: String] = [:]

    @ObservationIgnored
    let runtime: RuntimeServing?

    @ObservationIgnored
    private var eventTasks: [String: Task<Void, Never>] = [:]

    @ObservationIgnored
    private var healthMonitorTask: Task<Void, Never>?

    @ObservationIgnored
    var installMonitorTask: Task<Void, Never>?

    public init(runtime: RuntimeServing? = nil) {
        if let runtime {
            self.runtime = runtime
            return
        }
        do {
            let client = try RuntimeClient()
            self.runtime = client
            startHealthMonitor(client: client)
            startInstallMonitor(client: client)
        } catch {
            self.runtime = nil
            self.bootstrapError = error.localizedDescription
        }
    }

    deinit {
        eventTasks.values.forEach { $0.cancel() }
        healthMonitorTask?.cancel()
        installMonitorTask?.cancel()
    }

    // MARK: - Derived State

    public var libraryEntries: [LibraryEntry] {
        LibraryEntry.flatten(jobs: jobs)
    }

    public var activeJobCount: Int {
        jobs.filter { !$0.state.isTerminal }.count
    }

    public var hasBootstrapped: Bool {
        runtimeStatus != nil
    }

    public var hasPendingModelSetup: Bool {
        hasBootstrapped && !hasCompletedOnboarding && !catalog.recommendedAvailableItems.isEmpty && catalog.recommendedInstalledItems.isEmpty
    }

    public var activeInstallCount: Int {
        installOperations.filter { !$0.phase.isTerminal }.count
    }

    // MARK: - Refresh

    public func refresh() async {
        guard let runtime else {
            globalError = bootstrapError ?? "The MLXR runtime bridge is not available."
            return
        }
        isRefreshing = true
        defer { isRefreshing = false }

        do {
            async let status = runtime.status()
            async let supportedModels = runtime.listSupportedModels()
            async let installedModels = runtime.listModels()
            async let capabilities = runtime.listCapabilities()
            async let jobs = runtime.listJobs()
            async let installOperations = runtime.listModelInstalls()

            let (resolvedStatus, resolvedSupportedModels, resolvedInstalledModels, resolvedCapabilities, resolvedJobs, resolvedInstallOperations) = try await (
                status,
                supportedModels,
                installedModels,
                capabilities,
                jobs,
                installOperations
            )

            runtimeStatus = resolvedStatus
            catalog = CatalogSnapshot(
                supportedModels: resolvedSupportedModels,
                installedModels: resolvedInstalledModels,
                capabilities: resolvedCapabilities
            )
            self.jobs = resolvedJobs.sorted { $0.updatedAt > $1.updatedAt }
            self.installOperations = resolvedInstallOperations.sorted { $0.updatedAt > $1.updatedAt }
            globalError = nil
            bootstrapError = nil
            runtimeProcessDied = false
            if catalog.recommendedInstalledItems.isEmpty {
                await loadSetupPreviewsIfNeeded()
            } else {
                hasCompletedOnboarding = true
            }
        } catch {
            globalError = error.localizedDescription
        }
    }

    // MARK: - Error Dismissal

    public func dismissImageError() { imageError = nil }
    public func dismissVideoError() { videoError = nil }
    public func dismissModelsError() { modelsError = nil }
    public func dismissSettingsError() { settingsError = nil }
    public func dismissGlobalError() { globalError = nil }

    // MARK: - Image Submission

    public func submitImage(_ request: ImageGenerationRequest) async {
        imageError = nil
        isSubmittingImage = true
        defer { isSubmittingImage = false }
        do {
            let intent = try await buildImageIntent(from: request)
            try await submit(intent: intent)
        } catch {
            imageError = error.localizedDescription
        }
    }

    // MARK: - Video Submission

    public func submitVideo(_ request: VideoGenerationRequest) async {
        videoError = nil
        isSubmittingVideo = true
        defer { isSubmittingVideo = false }
        do {
            let intent = try await buildVideoIntent(from: request)
            try await submit(intent: intent)
        } catch {
            videoError = error.localizedDescription
        }
    }

    // MARK: - Job Management

    public func cancelJob(jobId: String) async {
        guard let runtime else { return }
        do {
            _ = try await runtime.cancel(jobId: jobId)
            await refreshJobsOnly()
        } catch {
            globalError = error.localizedDescription
        }
    }

    // MARK: - Output

    public func cachedOutputURL(for artifact: OutputArtifactRecord) async -> URL? {
        guard let runtime else { return nil }
        do {
            return try await runtime.cachedDownloadURL(for: artifact)
        } catch {
            globalError = error.localizedDescription
            return nil
        }
    }

    public func exportOutput(_ artifact: OutputArtifactRecord, to destination: URL) async {
        guard let runtime else { return }
        do {
            _ = try await runtime.exportOutput(
                artifactId: artifact.artifactId,
                destinationPath: destination,
                overwrite: true
            )
        } catch {
            globalError = error.localizedDescription
        }
    }

    // MARK: - Advanced Import

    public func inspectAdvancedImport(_ draft: AdvancedImportDraft) async throws -> [String: SourceInspectionResult] {
        guard let runtime else {
            throw RuntimeBridgeError.featureUnavailable("The runtime bridge is unavailable.")
        }
        let refs = try sourceRefs(for: draft)
        var results: [String: SourceInspectionResult] = [:]
        for (role, ref) in refs.sorted(by: { $0.key < $1.key }) {
            results[role] = try await runtime.inspectSource(ref)
        }
        return results
    }

    public func runAdvancedImport(_ draft: AdvancedImportDraft) async throws {
        guard let runtime else {
            throw RuntimeBridgeError.featureUnavailable("The runtime bridge is unavailable.")
        }
        let refs = try sourceRefs(for: draft)
        var sourceIds: [String: String] = [:]
        for (role, ref) in refs {
            let registration = try await runtime.registerSource(ref)
            sourceIds[role] = registration.sourceId
        }

        switch draft.mode {
        case .huggingFaceSingleRepo, .trustedLocalBundle:
            guard let sourceId = sourceIds["bundle"] else {
                throw RuntimeBridgeError.invalidResponse("Expected a bundle source registration.")
            }
            _ = try await runtime.convertArtifact(
                ArtifactConversionRequest(sourceId: sourceId, modelId: draft.modelId)
            )
        case .ltxMultiSource:
            _ = try await runtime.convertArtifact(
                ArtifactConversionRequest(
                    sourceBindings: sourceIds,
                    family: "ltx",
                    modelId: draft.modelId
                )
            )
        }
        await refresh()
    }

    // MARK: - Private: Submission

    private func submit(intent: WorkflowIntent) async throws {
        guard let runtime else {
            throw RuntimeBridgeError.featureUnavailable("The runtime bridge is unavailable.")
        }
        let result = try await runtime.run(intent: intent)
        await refreshJobsOnly()
        observe(jobId: result.submit.jobId)
    }

    private func refreshJobsOnly() async {
        guard let runtime else { return }
        do {
            jobs = try await runtime.listJobs().sorted { $0.updatedAt > $1.updatedAt }
        } catch {
            globalError = error.localizedDescription
        }
    }

    private func observe(jobId: String) {
        eventTasks[jobId]?.cancel()
        guard let runtime else { return }
        eventTasks[jobId] = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in runtime.streamEvents(jobId: jobId) {
                    await MainActor.run {
                        if let phase = event.phase {
                            self.activeJobPhases[jobId] = phase
                        }
                        if event.kind == .jobCompleted || event.kind == .jobFailed || event.kind == .jobCancelled {
                            self.activeJobPhases.removeValue(forKey: jobId)
                        }
                    }
                    await self.refreshJobsOnly()
                }
            } catch {
                _ = await MainActor.run {
                    self.activeJobPhases.removeValue(forKey: jobId)
                }
            }
        }
    }

    // MARK: - Private: Health Monitor

    private func startHealthMonitor(client: RuntimeClient) {
        healthMonitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
                guard let self else { return }
                let dead = await client.isRuntimeProcessDead
                await MainActor.run {
                    if dead && !self.runtimeProcessDied {
                        self.runtimeProcessDied = true
                        self.globalError = "The MLXR runtime process has terminated unexpectedly. Try refreshing to reconnect."
                    }
                }
            }
        }
    }

    // MARK: - Private: Intent Builders

    private func buildImageIntent(from request: ImageGenerationRequest) async throws -> WorkflowIntent {
        let references = try await importReferences(request.references)
        var params: JSONMap = [
            "width": .integer(request.width),
            "height": .integer(request.height),
            "num_inference_steps": .integer(request.numInferenceSteps),
            "guidance_scale": .number(request.guidanceScale),
        ]
        if let seed = request.seed {
            params["seed"] = .integer(seed)
        }
        return WorkflowIntent(
            modelId: request.modelId,
            prompt: request.prompt,
            task: request.task.rawValue,
            negativePrompt: request.negativePrompt.isEmpty ? nil : request.negativePrompt,
            references: references,
            params: params,
            output: JobOutputPolicy(artifactFormat: request.artifactFormat),
            preferences: WorkflowPreferences(quality: request.quality),
            extensions: namespacedExtensions(for: request.modelId, familyExtensions: request.familyExtensions)
        )
    }

    private func buildVideoIntent(from request: VideoGenerationRequest) async throws -> WorkflowIntent {
        let references = try await importReferences(request.references)
        var params: JSONMap = [
            "width": .integer(request.width),
            "height": .integer(request.height),
            "num_frames": .integer(request.numFrames),
            "fps": .integer(request.fps),
            "num_inference_steps": .integer(request.numInferenceSteps),
            "guidance_scale": .number(request.guidanceScale),
            "regenerate_video": .bool(request.regenerateVideo),
            "regenerate_audio": .bool(request.regenerateAudio),
        ]
        if let seed = request.seed {
            params["seed"] = .integer(seed)
        }
        if let windowStartSeconds = request.windowStartSeconds {
            params["window_start_seconds"] = .number(windowStartSeconds)
        }
        if let windowEndSeconds = request.windowEndSeconds {
            params["window_end_seconds"] = .number(windowEndSeconds)
        }

        var familyExtensions = request.familyExtensions
        if let workflowVariant = request.workflowVariant {
            familyExtensions["workflow_variant"] = .string(workflowVariant)
        }
        if let controlVariant = request.controlVariant {
            familyExtensions["control_variant"] = .string(controlVariant)
        }
        if let conditioningAttentionStrength = request.conditioningAttentionStrength {
            familyExtensions["conditioning_attention_strength"] = .number(conditioningAttentionStrength)
        }

        return WorkflowIntent(
            modelId: request.modelId,
            prompt: request.prompt,
            task: request.task.rawValue,
            negativePrompt: request.negativePrompt.isEmpty ? nil : request.negativePrompt,
            references: references,
            params: params,
            output: JobOutputPolicy(artifactFormat: request.artifactFormat),
            preferences: WorkflowPreferences(quality: request.quality),
            extensions: namespacedExtensions(for: request.modelId, familyExtensions: familyExtensions)
        )
    }

    private func importReferences(_ references: [MediaReferenceInput]) async throws -> [WorkflowReference] {
        guard let runtime else {
            throw RuntimeBridgeError.featureUnavailable("The runtime bridge is unavailable.")
        }
        var imported: [WorkflowReference] = []
        for reference in references {
            let handle = try await runtime.importFile(at: reference.fileURL, kind: reference.kind)
            imported.append(
                WorkflowReference(
                    inputHandle: handle.handleId,
                    kind: reference.kind,
                    role: reference.role,
                    metadata: reference.metadata
                )
            )
        }
        return imported
    }

    private func namespacedExtensions(for modelId: String, familyExtensions: JSONMap) -> JSONMap {
        guard !familyExtensions.isEmpty, let family = catalog.items.first(where: { $0.modelId == modelId })?.family else {
            return [:]
        }
        return [family: .object(familyExtensions)]
    }

    private func sourceRefs(for draft: AdvancedImportDraft) throws -> [String: SourceRef] {
        guard !draft.modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RuntimeBridgeError.featureUnavailable("Choose a model ID before importing.")
        }

        switch draft.mode {
        case .huggingFaceSingleRepo:
            guard !draft.huggingFaceRepo.isEmpty else {
                throw RuntimeBridgeError.featureUnavailable("Enter a Hugging Face repo first.")
            }
            return [
                "bundle": SourceRef(
                    provider: "huggingface",
                    locator: [
                        "repo": .string(draft.huggingFaceRepo),
                        "revision": .string("main"),
                    ],
                    auth: SourceAuth(tokenRef: "hf-default"),
                    familyHint: draft.familyHint
                )
            ]
        case .trustedLocalBundle:
            guard !draft.localPath.isEmpty else {
                throw RuntimeBridgeError.featureUnavailable("Choose a local bundle path first.")
            }
            return [
                "bundle": SourceRef(
                    provider: "local",
                    locator: ["path": .string(draft.localPath)],
                    familyHint: draft.familyHint
                )
            ]
        case .ltxMultiSource:
            guard !draft.ltxCheckpointRepo.isEmpty, !draft.ltxTextEncoderRepo.isEmpty else {
                throw RuntimeBridgeError.featureUnavailable("Checkpoint and text encoder repos are required for LTX import.")
            }
            var refs: [String: SourceRef] = [
                "checkpoint": .init(
                    provider: "huggingface",
                    locator: [
                        "repo": .string(draft.ltxCheckpointRepo),
                        "revision": .string("main"),
                        "role": .string("checkpoint"),
                    ],
                    auth: SourceAuth(tokenRef: "hf-default"),
                    familyHint: "ltx"
                ),
                "text_encoder": .init(
                    provider: "huggingface",
                    locator: [
                        "repo": .string(draft.ltxTextEncoderRepo),
                        "revision": .string("main"),
                        "role": .string("text_encoder"),
                    ],
                    auth: SourceAuth(tokenRef: "hf-default"),
                    familyHint: "ltx"
                ),
            ]
            if !draft.ltxUpsamplerRepo.isEmpty {
                refs["spatial_upsampler"] = .init(
                    provider: "huggingface",
                    locator: [
                        "repo": .string(draft.ltxUpsamplerRepo),
                        "revision": .string("main"),
                        "role": .string("spatial_upsampler"),
                    ],
                    auth: SourceAuth(tokenRef: "hf-default"),
                    familyHint: "ltx"
                )
            }
            if !draft.ltxDistilledLoRARepo.isEmpty {
                refs["distilled_lora"] = .init(
                    provider: "huggingface",
                    locator: [
                        "repo": .string(draft.ltxDistilledLoRARepo),
                        "revision": .string("main"),
                        "role": .string("distilled_lora"),
                    ],
                    auth: SourceAuth(tokenRef: "hf-default"),
                    familyHint: "ltx"
                )
            }
            return refs
        }
    }
}
