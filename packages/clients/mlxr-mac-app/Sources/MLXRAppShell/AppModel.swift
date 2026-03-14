import Foundation
import MLXRAppDomain
import MLXRRuntimeBridge
import Observation

private struct AppRefreshSnapshot {
    let status: RuntimeStatusSnapshot
    let supportedModels: [SupportedModelDescriptor]
    let installedModels: [ModelRecord]
    let capabilities: [CapabilityDescriptor]
    let jobs: [JobRecord]
    let installOperations: [ModelInstallOperationRecord]
}

private struct CatalogRefreshSnapshot {
    let supportedModels: [SupportedModelDescriptor]
    let installedModels: [ModelRecord]
    let capabilities: [CapabilityDescriptor]
}

@MainActor
@Observable
public final class MLXRAppModel {
    private static let minimumFullRefreshInterval: TimeInterval = 1.5
    private static let minimumJobsRefreshInterval: TimeInterval = 1
    private static let deferredJobsRefreshIntervalNanoseconds: UInt64 = 750_000_000
    private static let healthMonitorIntervalNanoseconds: UInt64 = 15_000_000_000

    // -- State --
    public var runtimeStatus: RuntimeStatusSnapshot?
    public var catalog = CatalogSnapshot(supportedModels: [], installedModels: [], capabilities: [])
    public var jobs: [JobRecord] = []
    public var importedAssets: [ImportedAssetRecord] = []
    public var installOperations: [ModelInstallOperationRecord] = []
    public var modelPreviews: [String: SupportedModelPreview] = [:]
    public var installedModelDetails: [String: InstalledModelDetails] = [:]
    public var lastSubmittedJobId: String?
    public var isRefreshing = false
    public var isSubmittingImage = false
    public var isSubmittingVideo = false
    public var isPlanningStudio = false
    public var promptHelperMode: PromptHelperMode = .suggest
    public var hasCompletedOnboarding = false {
        didSet { persistPresentationState() }
    }
    public var activeWorkspaceId = "default-workspace" {
        didSet { persistPresentationState() }
    }
    public var selectedLibraryWorkspaceId: String? {
        didSet { persistPresentationState() }
    }
    public var workspaces: [WorkspaceRecord] = [WorkspaceRecord(id: "default-workspace", title: "New Project")] {
        didSet { persistPresentationState() }
    }
    public var collections: [CollectionRecord] = [] {
        didSet { persistPresentationState() }
    }
    public var runGroups: [RunGroupRecord] = [] {
        didSet { persistPresentationState() }
    }
    public var dismissedActivityRunGroupIds: Set<String> = [] {
        didSet { persistPresentationState() }
    }
    public var assetRecords: [String: AssetRecord] = [:] {
        didSet { persistPresentationState() }
    }
    public var studioWorkspace = StudioWorkspaceDraft() {
        didSet { persistPresentationState() }
    }

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
    public var studioPlanResult: WorkflowPlanResult?
    public var studioPlanError: String?

    @ObservationIgnored
    let runtime: RuntimeServing?

    @ObservationIgnored
    let importedAssetStore: ImportedAssetStore

    @ObservationIgnored
    let workspaceStateStore: WorkspaceStateStore

    @ObservationIgnored
    private var eventTasks: [String: Task<Void, Never>] = [:]

    @ObservationIgnored
    private var healthMonitorTask: Task<Void, Never>?

    @ObservationIgnored
    var installMonitorTask: Task<Void, Never>?

    @ObservationIgnored
    private var refreshTask: Task<AppRefreshSnapshot, Error>?

    @ObservationIgnored
    private var jobsRefreshTask: Task<Void, Never>?

    @ObservationIgnored
    private var studioPlanTask: Task<Void, Never>?

    @ObservationIgnored
    private var lastFullRefreshAt: Date?

    @ObservationIgnored
    private var lastJobsRefreshAt: Date?

    @ObservationIgnored
    private var observedJobIds: Set<String> = []

    public init(runtime: RuntimeServing? = nil) {
        let store = ImportedAssetStore()
        self.importedAssetStore = store
        let workspaceStateStore = WorkspaceStateStore()
        self.workspaceStateStore = workspaceStateStore
        do {
            importedAssets = try store.load()
        } catch {
            importedAssets = []
            settingsError = error.localizedDescription
        }
        do {
            let presentation = try workspaceStateStore.load()
            hasCompletedOnboarding = presentation.hasCompletedModelSetup
            activeWorkspaceId = presentation.activeWorkspaceId
            selectedLibraryWorkspaceId = presentation.selectedLibraryWorkspaceId
            workspaces = presentation.workspaces
            collections = presentation.collections
            runGroups = presentation.runGroups
            dismissedActivityRunGroupIds = Set(presentation.dismissedActivityRunGroupIds)
            assetRecords = Dictionary(uniqueKeysWithValues: presentation.assets.map { ($0.id, $0) })
            studioWorkspace = presentation.workspaceDraft
        } catch {
            settingsError = error.localizedDescription
        }
        if let runtime {
            self.runtime = runtime
            return
        }
        do {
            let client = try RuntimeClient()
            self.runtime = client
            startHealthMonitor(client: client)
        } catch {
            self.runtime = nil
            self.bootstrapError = error.localizedDescription
        }
    }

    deinit {
        eventTasks.values.forEach { $0.cancel() }
        healthMonitorTask?.cancel()
        installMonitorTask?.cancel()
        jobsRefreshTask?.cancel()
        studioPlanTask?.cancel()
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

    public var hasWorkspaceDraft: Bool {
        studioWorkspace.hasMeaningfulState
    }

    public var activeWorkspace: WorkspaceRecord? {
        workspaces.first(where: { $0.id == activeWorkspaceId })
    }

    // MARK: - Refresh

    public func refresh() async {
        await performFullRefresh(force: false)
    }

    func performFullRefresh(force: Bool) async {
        guard let runtime else {
            globalError = bootstrapError ?? "The MLXR runtime bridge is not available."
            return
        }
        if let refreshTask {
            _ = try? await refreshTask.value
            return
        }
        if
            !force,
            let lastFullRefreshAt,
            Date().timeIntervalSince(lastFullRefreshAt) < Self.minimumFullRefreshInterval
        {
            return
        }

        isRefreshing = true
        let task = Task<AppRefreshSnapshot, Error> {
            async let status = runtime.status()
            async let supportedModels = runtime.listSupportedModels()
            async let installedModels = runtime.listModels()
            async let capabilities = runtime.listCapabilities()
            async let jobs = runtime.listJobs()
            async let installOperations = runtime.listModelInstalls()

            let (
                resolvedStatus,
                resolvedSupportedModels,
                resolvedInstalledModels,
                resolvedCapabilities,
                resolvedJobs,
                resolvedInstallOperations
            ) = try await (
                status,
                supportedModels,
                installedModels,
                capabilities,
                jobs,
                installOperations
            )

            return AppRefreshSnapshot(
                status: resolvedStatus,
                supportedModels: resolvedSupportedModels,
                installedModels: resolvedInstalledModels,
                capabilities: resolvedCapabilities,
                jobs: resolvedJobs.sorted { $0.updatedAt > $1.updatedAt },
                installOperations: resolvedInstallOperations.sorted { $0.updatedAt > $1.updatedAt }
            )
        }
        refreshTask = task
        defer {
            refreshTask = nil
            isRefreshing = false
        }

        do {
            let snapshot = try await task.value
            runtimeStatus = snapshot.status
            applyCatalogSnapshot(
                CatalogRefreshSnapshot(
                    supportedModels: snapshot.supportedModels,
                    installedModels: snapshot.installedModels,
                    capabilities: snapshot.capabilities
                )
            )
            jobs = snapshot.jobs
            syncObservedJobs()
            syncRunGroupsFromJobs()
            installOperations = snapshot.installOperations
            syncInstallMonitor()
            globalError = nil
            bootstrapError = nil
            runtimeProcessDied = false
            lastFullRefreshAt = Date()
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
    public func dismissStudioPlanError() { studioPlanError = nil }

    // MARK: - Studio Planning

    public func scheduleStudioPlan() {
        studioPlanTask?.cancel()
        studioPlanTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 200_000_000)
            guard let self else { return }
            await self.refreshStudioPlan()
        }
    }

    public func refreshStudioPlan() async {
        guard let runtime else {
            studioPlanResult = nil
            studioPlanError = bootstrapError
            return
        }
        guard let intent = studioPlanningIntent() else {
            studioPlanTask = nil
            isPlanningStudio = false
            studioPlanResult = nil
            studioPlanError = nil
            return
        }
        isPlanningStudio = true
        defer {
            isPlanningStudio = false
            studioPlanTask = nil
        }

        do {
            studioPlanResult = try await runtime.plan(intent: intent)
            studioPlanError = nil
        } catch is CancellationError {
            return
        } catch {
            studioPlanResult = nil
            studioPlanError = error.localizedDescription
        }
    }

    // MARK: - Image Submission

    public func submitImage(_ request: ImageGenerationRequest) async {
        imageError = nil
        isSubmittingImage = true
        defer { isSubmittingImage = false }
        do {
            let intent = try await buildImageIntent(from: request)
            try await submit(
                intent: intent,
                task: request.task,
                runGroupTitle: request.runGroupTitle,
                variationCount: request.variationCount
            )
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
            try await submit(
                intent: intent,
                task: request.task,
                runGroupTitle: request.runGroupTitle,
                variationCount: request.variationCount
            )
        } catch {
            videoError = error.localizedDescription
        }
    }

    // MARK: - Job Management

    public func cancelJob(jobId: String) async {
        guard let runtime else { return }
        do {
            _ = try await runtime.cancel(jobId: jobId)
            await refreshJobsOnly(force: true)
        } catch {
            globalError = error.localizedDescription
        }
    }

    public func cancelRunGroup(runGroupId: String) async {
        let activeJobs: [JobRecord]
        if let legacyJobId = runGroupId.split(separator: "-", maxSplits: 1).last, runGroupId.hasPrefix("legacy-") {
            activeJobs = jobs.filter { $0.jobId == String(legacyJobId) && !$0.state.isTerminal }
        } else {
            activeJobs = jobs.filter {
                $0.request.context?.runGroupId == runGroupId && !$0.state.isTerminal
            }
        }
        guard !activeJobs.isEmpty else { return }
        for job in activeJobs {
            await cancelJob(jobId: job.jobId)
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

    private func submit(
        intent: WorkflowIntent,
        task: ProductTask,
        runGroupTitle: String,
        variationCount: Int
    ) async throws {
        guard let runtime else {
            throw RuntimeBridgeError.featureUnavailable("The runtime bridge is unavailable.")
        }
        let result = try await runtime.run(intent: intent)
        if let runGroupId = intent.context?.runGroupId {
            recordAcceptedRunGroup(
                runGroupId: runGroupId,
                task: task,
                title: runGroupTitle,
                context: intent.context,
                jobId: result.submit.jobId,
                variationCount: variationCount
            )
        } else {
            lastSubmittedJobId = result.submit.jobId
        }
        await refreshJobsOnly(force: true)
        observe(jobId: result.submit.jobId)
    }

    func refreshCatalogOnly(force _: Bool = false) async {
        guard let runtime else { return }
        if let refreshTask {
            _ = try? await refreshTask.value
            return
        }

        do {
            async let supportedModels = runtime.listSupportedModels()
            async let installedModels = runtime.listModels()
            async let capabilities = runtime.listCapabilities()

            let snapshot = try await CatalogRefreshSnapshot(
                supportedModels: supportedModels,
                installedModels: installedModels,
                capabilities: capabilities
            )
            applyCatalogSnapshot(snapshot)
            modelsError = nil
        } catch {
            modelsError = error.localizedDescription
        }
    }

    private func applyCatalogSnapshot(_ snapshot: CatalogRefreshSnapshot) {
        catalog = CatalogSnapshot(
            supportedModels: snapshot.supportedModels,
            installedModels: snapshot.installedModels,
            capabilities: snapshot.capabilities
        )
        if catalog.recommendedInstalledItems.isEmpty {
            Task { await self.loadSetupPreviewsIfNeeded() }
        } else {
            hasCompletedOnboarding = true
        }
    }

    private func refreshJobsOnly(force: Bool = false) async {
        guard let runtime else { return }
        if
            !force,
            let lastJobsRefreshAt,
            Date().timeIntervalSince(lastJobsRefreshAt) < Self.minimumJobsRefreshInterval
        {
            return
        }

        do {
            jobs = try await runtime.listJobs().sorted { $0.updatedAt > $1.updatedAt }
            lastJobsRefreshAt = Date()
            syncObservedJobs()
            syncRunGroupsFromJobs()
        } catch {
            globalError = error.localizedDescription
        }
    }

    private func scheduleJobsRefresh(immediate: Bool) {
        if immediate {
            jobsRefreshTask?.cancel()
        } else if jobsRefreshTask != nil {
            return
        }

        jobsRefreshTask = Task { [weak self] in
            if !immediate {
                try? await Task.sleep(nanoseconds: Self.deferredJobsRefreshIntervalNanoseconds)
            }
            guard let self else { return }
            await self.refreshJobsOnly(force: immediate)
            await MainActor.run {
                self.jobsRefreshTask = nil
            }
        }
    }

    private func observe(jobId: String) {
        guard let runtime else { return }
        guard eventTasks[jobId] == nil else { return }
        eventTasks[jobId] = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in runtime.streamEvents(jobId: jobId) {
                    await MainActor.run {
                        if let phase = event.phase {
                            self.activeJobPhases[jobId] = phase
                        }
                        let isTerminalEvent =
                            event.kind == .jobCompleted
                            || event.kind == .jobFailed
                            || event.kind == .jobCancelled
                        if isTerminalEvent {
                            self.activeJobPhases.removeValue(forKey: jobId)
                        }
                        if isTerminalEvent || event.kind == .jobArtifactReady {
                            self.scheduleJobsRefresh(immediate: true)
                        } else if event.kind == .jobAccepted || event.kind == .jobPhaseChanged {
                            self.scheduleJobsRefresh(immediate: false)
                        }
                    }
                }
            } catch {
                _ = await MainActor.run {
                    self.activeJobPhases.removeValue(forKey: jobId)
                }
            }
            await MainActor.run {
                self.eventTasks.removeValue(forKey: jobId)
                self.observedJobIds.remove(jobId)
            }
        }
    }

    // MARK: - Private: Health Monitor

    private func startHealthMonitor(client: RuntimeClient) {
        healthMonitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.healthMonitorIntervalNanoseconds)
                guard let self else { return }
                let shouldCheck = await MainActor.run {
                    self.runtimeStatus?.launchedByApp == true
                }
                if !shouldCheck {
                    continue
                }
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
            context: request.context,
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
            context: request.context,
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

    func namespacedExtensions(for modelId: String, familyExtensions: JSONMap) -> JSONMap {
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

    private func syncObservedJobs() {
        let activeIds = Set(jobs.filter { !$0.state.isTerminal }.map(\.jobId))

        for jobId in observedJobIds.subtracting(activeIds) {
            eventTasks[jobId]?.cancel()
            eventTasks.removeValue(forKey: jobId)
            activeJobPhases.removeValue(forKey: jobId)
        }

        for jobId in activeIds.subtracting(observedJobIds) {
            observe(jobId: jobId)
        }

        observedJobIds = activeIds
    }
}
