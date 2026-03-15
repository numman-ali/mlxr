import Foundation
import Testing
@testable import MLXRAppDomain
@testable import MLXRAppShell
@testable import MLXRRuntimeBridge

// MARK: - Mock Runtime

final class MockRuntime: RuntimeServing, @unchecked Sendable {
    var statusResult: RuntimeStatusSnapshot?
    var statusError: Error?
    var supportedModels: [SupportedModelDescriptor] = []
    var installedModels: [ModelRecord] = []
    var capabilities: [CapabilityDescriptor] = []
    var jobs: [JobRecord] = []
    var installOperations: [ModelInstallOperationRecord] = []
    var supportedModelPreviews: [String: SupportedModelPreview] = [:]
    var modelDetailsById: [String: InstalledModelDetails] = [:]
    var installResult: ModelInstallResult?
    var installError: Error?
    var enqueueInstallResult: ModelInstallOperationRecord?
    var enqueueInstallError: Error?
    var cancelInstallResult: ModelInstallOperationRecord?
    var removeModelResult: ModelRemoveResult?
    var runResult: WorkflowRunResult?
    var runError: Error?

    var submitCallCount = 0
    var installCallCount = 0
    var enqueueInstallCallCount = 0
    var removeModelCallCount = 0
    var cancelCallCount = 0

    func status() async throws -> RuntimeStatusSnapshot {
        if let error = statusError { throw error }
        return statusResult ?? RuntimeStatusSnapshot(
            health: RuntimeHealth(status: "ok", transportDefault: "uds"),
            runtimeHome: URL(fileURLWithPath: "/tmp/test-runtime"),
            socketPath: URL(fileURLWithPath: "/tmp/test.sock"),
            logFile: URL(fileURLWithPath: "/tmp/test.log"),
            launchedByApp: false
        )
    }

    func listSupportedModels() async throws -> [SupportedModelDescriptor] { supportedModels }
    func previewSupportedModel(modelId: String) async throws -> SupportedModelPreview {
        guard let preview = supportedModelPreviews[modelId] else {
            throw RuntimeBridgeError.featureUnavailable("Preview missing in mock")
        }
        return preview
    }
    func listModels() async throws -> [ModelRecord] { installedModels }
    func modelDetails(modelId: String) async throws -> InstalledModelDetails {
        guard let details = modelDetailsById[modelId] else {
            throw RuntimeBridgeError.featureUnavailable("Details missing in mock")
        }
        return details
    }
    func listCapabilities() async throws -> [CapabilityDescriptor] { capabilities }
    func listJobs() async throws -> [JobRecord] { jobs }
    func listModelInstalls() async throws -> [ModelInstallOperationRecord] { installOperations }

    func installModel(modelId: String) async throws -> ModelInstallResult {
        installCallCount += 1
        if let error = installError { throw error }
        return installResult!
    }

    func enqueueModelInstall(modelId: String) async throws -> ModelInstallOperationRecord {
        enqueueInstallCallCount += 1
        if let error = enqueueInstallError { throw error }
        return enqueueInstallResult!
    }

    func cancelModelInstall(operationId: String) async throws -> ModelInstallOperationRecord {
        guard let cancelInstallResult else {
            throw RuntimeBridgeError.featureUnavailable("Cancel result missing in mock")
        }
        return cancelInstallResult
    }

    func removeModel(modelId: String) async throws -> ModelRemoveResult {
        removeModelCallCount += 1
        guard let removeModelResult else {
            throw RuntimeBridgeError.featureUnavailable("Remove result missing in mock")
        }
        return removeModelResult
    }

    func inspectSource(_ sourceRef: SourceRef) async throws -> SourceInspectionResult {
        throw RuntimeBridgeError.featureUnavailable("Not implemented in mock")
    }

    func registerSource(_ sourceRef: SourceRef) async throws -> SourceRegistrationRecord {
        throw RuntimeBridgeError.featureUnavailable("Not implemented in mock")
    }

    func convertArtifact(_ request: ArtifactConversionRequest) async throws -> ArtifactConversionResult {
        throw RuntimeBridgeError.featureUnavailable("Not implemented in mock")
    }

    func importFile(at fileURL: URL, kind: WorkflowReferenceKind) async throws -> InputHandleRecord {
        InputHandleRecord(
            handleId: "test-handle",
            mediaType: "image/png",
            role: kind.rawValue,
            metadata: [:],
            filename: fileURL.lastPathComponent,
            sizeBytes: 1024,
            storageKey: "test-key",
            createdAt: Date()
        )
    }

    func plan(intent: WorkflowIntent) async throws -> WorkflowPlanResult {
        throw RuntimeBridgeError.featureUnavailable("Not implemented in mock")
    }

    func run(intent: WorkflowIntent) async throws -> WorkflowRunResult {
        submitCallCount += 1
        if let error = runError { throw error }
        return runResult!
    }

    func cancel(jobId: String) async throws -> JobRecord {
        cancelCallCount += 1
        return JobRecord(
            jobId: jobId,
            request: JobRequest(modelId: "test", task: "test", inputs: [:], params: [:], output: .init(), extensions: [:]),
            state: .cancelled,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            artifacts: []
        )
    }

    func streamEvents(jobId: String) -> AsyncThrowingStream<RuntimeEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func exportOutput(artifactId: String, destinationPath: URL, overwrite: Bool) async throws -> ArtifactExportResult {
        ArtifactExportResult(artifactId: artifactId, destinationPath: destinationPath.path, sizeBytes: 0)
    }

    func cachedDownloadURL(for artifact: OutputArtifactRecord) async throws -> URL {
        URL(fileURLWithPath: "/tmp/cached-\(artifact.artifactId)")
    }
}

// MARK: - Tests

@Test
@MainActor
func refreshPopulatesCatalogAndJobs() async {
    let mock = MockRuntime()
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "test-model",
            displayName: "Test Model",
            family: "test",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "test",
            sourceSummary: "test/test-model",
            license: "mit",
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        )
    ]
    mock.installOperations = [
        ModelInstallOperationRecord(
            operationId: "mdl_1",
            modelId: "test-model",
            phase: .queued,
            supportedModel: mock.supportedModels[0],
            preview: nil,
            result: nil,
            error: nil,
            createdAt: Date(),
            updatedAt: Date(),
            startedAt: nil,
            finishedAt: nil,
            metadata: [:]
        )
    ]
    mock.jobs = [
        JobRecord(
            jobId: "job-1",
            request: JobRequest(modelId: "test-model", task: "image.generate", inputs: ["prompt": .string("hello")], params: [:], output: .init(), extensions: [:]),
            state: .completed,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            artifacts: []
        )
    ]

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.catalog.items.count == 1)
    #expect(model.catalog.items[0].modelId == "test-model")
    #expect(model.jobs.count == 1)
    #expect(model.installOperations.count == 1)
    #expect(model.globalError == nil)
    #expect(model.hasBootstrapped)
}

@Test
@MainActor
func refreshSurfacesErrorOnFailure() async {
    let mock = MockRuntime()
    mock.statusError = RuntimeBridgeError.runtimeUnavailable("Connection refused")

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.globalError != nil)
    #expect(model.globalError!.contains("Connection refused"))
}

@Test
@MainActor
func submitImageSurfacesErrorToImageError() async {
    let mock = MockRuntime()
    mock.runError = RuntimeBridgeError.api(statusCode: 500, detail: "Internal error")

    let model = MLXRAppModel(runtime: mock)
    // Need catalog populated to resolve family
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "test-img",
            displayName: "Test Image",
            family: "test",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "test",
            sourceSummary: "test",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        )
    ]
    await model.refresh()

    await model.submitImage(ImageGenerationRequest(
        task: .imageGenerate,
        modelId: "test-img",
        prompt: "test prompt"
    ))

    #expect(model.imageError != nil)
    #expect(model.globalError == nil, "Image errors should surface to imageError, not globalError")
}

@Test
@MainActor
func submitVideoSurfacesErrorToVideoError() async {
    let mock = MockRuntime()
    mock.runError = RuntimeBridgeError.api(statusCode: 422, detail: "Bad request")

    let model = MLXRAppModel(runtime: mock)
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "test-vid",
            displayName: "Test Video",
            family: "ltx",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["video.generate"],
            provider: "test",
            sourceSummary: "test",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        )
    ]
    await model.refresh()

    await model.submitVideo(VideoGenerationRequest(
        task: .videoGenerate,
        modelId: "test-vid",
        prompt: "test prompt"
    ))

    #expect(model.videoError != nil)
    #expect(model.globalError == nil)
}

@Test
@MainActor
func submitCurrentWorkspaceReturnsAcceptedWorkspaceId() async {
    let now = Date(timeIntervalSince1970: 5_000)
    let capability = CapabilityDescriptor(
        modelId: "test-img",
        artifactDigest: "artifact",
        family: "test",
        familyVariant: nil,
        tasks: ["image.generate"],
        modalitiesIn: ["text"],
        modalitiesOut: ["image"],
        constraints: [:],
        conditioning: [:],
        profilesByTask: [:],
        streaming: [:],
        artifactsOut: ["image/png"],
        schedulerClass: "test.scheduler",
        hardwareTiers: [],
        dependencies: [:],
        policy: PolicyDescriptor(
            license: nil,
            accessState: "public",
            remoteCodeRequired: false,
            remoteCodeApproved: false,
            redistributionState: nil
        ),
        extensionsSchema: nil,
        metadata: [:]
    )
    let plan = WorkflowPlan(
        modelId: "test-img",
        family: "test",
        schedulerClass: "test.scheduler",
        selectedTask: "image.generate",
        selectedProfile: nil,
        pipelineVariant: nil,
        resolvedPrompt: "ship it",
        references: [],
        stages: [],
        warnings: [],
        metadata: [:]
    )
    let submittedRecord = JobRecord(
        jobId: "job-accepted",
        request: JobRequest(
            modelId: "test-img",
            task: "image.generate",
            inputs: ["prompt": .string("ship it")],
            params: [:],
            output: .init(),
            context: WorkflowContextMetadata(
                workspaceId: "workspace-accepted",
                runGroupId: "run-group-1",
                intentLabel: "ship it"
            ),
            extensions: [:]
        ),
        state: .accepted,
        createdAt: now,
        updatedAt: now,
        error: nil,
        artifacts: []
    )

    let mock = MockRuntime()
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "test-img",
            displayName: "Test Image",
            family: "test",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "test",
            sourceSummary: "test",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        )
    ]
    mock.capabilities = [capability]
    mock.runResult = WorkflowRunResult(
        plan: plan,
        submit: WorkflowRunSubmitResult(jobId: "job-accepted", record: submittedRecord)
    )

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()
    model.creationDraft.workspaceId = "workspace-accepted"
    model.creationDraft.prompt = "ship it"
    model.creationPlanResult = WorkflowPlanResult(
        capability: capability,
        plan: plan,
        readiness: WorkflowPlanReadiness(ready: true),
        presentation: WorkflowPlanPresentation(
            primaryMode: "image",
            selectedTask: "image.generate"
        )
    )

    let acceptedWorkspaceId = await model.submitCurrentWorkspace()

    #expect(acceptedWorkspaceId == "workspace-accepted")
    #expect(mock.submitCallCount == 1)
}

@Test
@MainActor
func dismissingActivityFailureHidesItWithoutAffectingRunningWork() {
    let model = MLXRAppModel(runtime: MockRuntime())
    let failedGroup = RunGroupRecord(
        id: "failed-group",
        workspaceId: "default-workspace",
        task: .imageGenerate,
        title: "Failed image",
        updatedAt: Date(),
        state: .failed
    )
    let runningGroup = RunGroupRecord(
        id: "running-group",
        workspaceId: "default-workspace",
        task: .imageGenerate,
        title: "Running image",
        updatedAt: Date(),
        state: .running
    )
    model.runGroups = [failedGroup, runningGroup]

    #expect(model.activityRunGroups.map(\.id).sorted() == ["failed-group", "running-group"])

    model.dismissActivityRunGroup("failed-group")

    #expect(model.activityRunGroups.map(\.id) == ["running-group"])
}

@Test
@MainActor
func activityRunGroupsKeepOnlyActiveOrFailedWork() {
    let model = MLXRAppModel(runtime: MockRuntime())
    let oldCompletedGroup = RunGroupRecord(
        id: "old-complete",
        workspaceId: "default-workspace",
        task: .imageGenerate,
        title: "Old completion",
        updatedAt: Date().addingTimeInterval(-(72 * 60 * 60)),
        state: .completed
    )
    let recentCompletedGroup = RunGroupRecord(
        id: "recent-complete",
        workspaceId: "default-workspace",
        task: .imageGenerate,
        title: "Recent completion",
        updatedAt: Date(),
        state: .completed
    )
    model.runGroups = [oldCompletedGroup, recentCompletedGroup]

    #expect(model.activityRunGroups.isEmpty)
}

@Test
@MainActor
func dismissErrorsClearsCorrectSurface() async {
    let mock = MockRuntime()
    let model = MLXRAppModel(runtime: mock)

    // Simulate errors on multiple surfaces
    model.imageError = "Image failed"
    model.videoError = "Video failed"
    model.globalError = "Global failed"

    model.dismissImageError()
    #expect(model.imageError == nil)
    #expect(model.videoError != nil)
    #expect(model.globalError != nil)

    model.dismissVideoError()
    #expect(model.videoError == nil)

    model.dismissGlobalError()
    #expect(model.globalError == nil)
}

@Test
@MainActor
func activeJobCountReflectsNonTerminalJobs() async {
    let mock = MockRuntime()
    mock.jobs = [
        JobRecord(
            jobId: "j1",
            request: JobRequest(modelId: "m", task: "t", inputs: [:], params: [:], output: .init(), extensions: [:]),
            state: .running,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            artifacts: []
        ),
        JobRecord(
            jobId: "j2",
            request: JobRequest(modelId: "m", task: "t", inputs: [:], params: [:], output: .init(), extensions: [:]),
            state: .completed,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            artifacts: []
        ),
        JobRecord(
            jobId: "j3",
            request: JobRequest(modelId: "m", task: "t", inputs: [:], params: [:], output: .init(), extensions: [:]),
            state: .preparing,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            artifacts: []
        ),
    ]

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.activeJobCount == 2, "Only non-terminal jobs should count as active")
}

@Test
@MainActor
func modelWithNoRuntimeSetsBootstrapError() async {
    let originalRepoRoot = ProcessInfo.processInfo.environment["MLXR_MAC_APP_REPO_ROOT"]
    setenv("MLXR_MAC_APP_REPO_ROOT", "/definitely-not-a-real-mlxr-repo", 1)
    defer {
        if let originalRepoRoot {
            setenv("MLXR_MAC_APP_REPO_ROOT", originalRepoRoot, 1)
        } else {
            unsetenv("MLXR_MAC_APP_REPO_ROOT")
        }
    }

    let model = MLXRAppModel(runtime: nil)

    await model.refresh()

    #expect(model.globalError != nil, "Should surface an error when runtime is nil")
}

@Test
@MainActor
func queueModelInstallTracksOperationWithoutBlockingGeneration() async {
    let mock = MockRuntime()
    let supportedModel = SupportedModelDescriptor(
        modelId: "flux2-klein-9b-local",
        displayName: "FLUX.2 Klein 9B",
        family: "flux2",
        familyVariant: "flux.2-klein-9b",
        recommendationTier: .recommended,
        supportLevel: .promoted,
        tasks: ["image.generate"],
        provider: "huggingface",
        sourceSummary: "black-forest-labs/FLUX.2-klein-9B",
        license: nil,
        accessState: "public",
        installed: false,
        installable: true,
        notes: nil
    )
    mock.enqueueInstallResult = ModelInstallOperationRecord(
        operationId: "mdl_install",
        modelId: supportedModel.modelId,
        phase: .queued,
        supportedModel: supportedModel,
        preview: nil,
        result: nil,
        error: nil,
        createdAt: Date(),
        updatedAt: Date(),
        startedAt: nil,
        finishedAt: nil,
        metadata: [:]
    )

    let model = MLXRAppModel(runtime: mock)
    await model.queueModelInstall(modelId: supportedModel.modelId)

    #expect(model.installOperations.count == 1)
    #expect(model.installOperations[0].modelId == supportedModel.modelId)
    #expect(model.isSubmittingImage == false)
    #expect(model.isSubmittingVideo == false)
    #expect(mock.enqueueInstallCallCount == 1)
}

@Test
@MainActor
func pendingModelSetupRequiresRecommendedInstallGap() async {
    let mock = MockRuntime()
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "qwen-image-local",
            displayName: "Qwen Image",
            family: "qwen_image",
            familyVariant: "qwen-image-2512",
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "huggingface",
            sourceSummary: "Qwen/Qwen-Image-2512",
            license: "apache-2.0",
            accessState: "public",
            installed: false,
            installable: true,
            notes: nil
        )
    ]
    mock.supportedModelPreviews["qwen-image-local"] = SupportedModelPreview(
        supportedModel: mock.supportedModels[0],
        sources: [],
        totalSourceBytes: 1024,
        authRequired: false,
        authMessage: nil
    )

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.hasPendingModelSetup)

    model.completeModelSetup()

    #expect(model.hasPendingModelSetup == false)
}

@Test
@MainActor
func removeInstalledModelClearsCachedDetailsAfterSuccess() async {
    let mock = MockRuntime()
    mock.removeModelResult = ModelRemoveResult(
        status: .removed,
        modelId: "z-image-turbo-local",
        artifactDigest: "sha256:test",
        removedSourceIds: ["src_1"],
        removedStorageKey: "artifacts-portable/z_image/z-image-turbo-local/sha256_test"
    )
    mock.modelDetailsById["z-image-turbo-local"] = InstalledModelDetails(
        model: ModelRecord(
            modelId: "z-image-turbo-local",
            family: "z_image",
            source: nil,
            artifact: nil,
            loaded: false,
            capability: nil
        ),
        supportedModel: nil,
        managedStorageKey: "artifacts-portable/z_image/z-image-turbo-local/sha256_test",
        managedSizeBytes: 2048,
        referencedSourceIds: ["src_1"]
    )

    let model = MLXRAppModel(runtime: mock)
    model.installedModelDetails["z-image-turbo-local"] = mock.modelDetailsById["z-image-turbo-local"]

    await model.removeInstalledModel(modelId: "z-image-turbo-local")

    #expect(model.installedModelDetails["z-image-turbo-local"] == nil)
    #expect(mock.removeModelCallCount == 1)
}

@Test
@MainActor
func defaultImageModelPrefersZImageTurboWhenAvailable() async {
    let mock = MockRuntime()
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "z-image-turbo-local",
            displayName: "Z-Image Turbo",
            family: "z_image",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "huggingface",
            sourceSummary: "Tongyi-MAI/Z-Image-Turbo",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        ),
        SupportedModelDescriptor(
            modelId: "qwen-image-local",
            displayName: "Qwen Image",
            family: "qwen_image",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "huggingface",
            sourceSummary: "Qwen/Qwen-Image-2512",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        ),
    ]

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.preferredDefaultModelId(for: .imageGenerate) == "z-image-turbo-local")
}

@Test
@MainActor
func seedComposerRequestSwitchesWorkspaceAndFocusesAsset() {
    let model = MLXRAppModel(runtime: MockRuntime())
    let targetWorkspace = WorkspaceRecord(id: "project-2", title: "Project Two")
    model.workspaces = [
        WorkspaceRecord(id: "default-workspace", title: "New Project"),
        targetWorkspace,
    ]
    model.importedAssets = [
        ImportedAssetRecord(
            id: "asset-1",
            workspaceId: targetWorkspace.id,
            title: "reference.png",
            sourcePath: "/tmp/reference.png",
            storageKey: "imports/reference.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        )
    ]
    model.creationDraft.prompt = "old prompt"
    model.creationDraft.referenceAssetIds = ["old-asset"]
    model.creationDraft.selectedAssetId = "old-asset"

    model.seedComposer(
        with: ComposerSeedRequest(
            workspaceId: targetWorkspace.id,
            task: .imageEdit,
            prompt: "keep the violinist identity",
            focusedAssetId: "asset-1",
            referenceAssetIds: ["asset-1"]
        )
    )

    #expect(model.activeWorkspaceId == targetWorkspace.id)
    #expect(model.selectedLibraryWorkspaceId == targetWorkspace.id)
    #expect(model.creationDraft.workspaceId == targetWorkspace.id)
    #expect(model.creationDraft.task == .imageEdit)
    #expect(model.creationDraft.prompt == "keep the violinist identity")
    #expect(model.creationDraft.selectedAssetId == "asset-1")
    #expect(model.creationDraft.preferredDisplayedAssetId == "asset-1")
    #expect(model.creationDraft.referenceAssetIds == ["asset-1"])
    #expect(model.assetRecords["asset-1"]?.lastUsedAt != nil)
}

@Test
@MainActor
func seedComposerFallsBackToFocusedAssetTitleWhenPromptIsEmpty() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.importedAssets = [
        ImportedAssetRecord(
            id: "asset-1",
            workspaceId: "default-workspace",
            title: "reference.png",
            sourcePath: "/tmp/reference.png",
            storageKey: "imports/reference.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        )
    ]
    model.creationDraft.prompt = ""

    model.seedComposer(
        with: ComposerSeedRequest(
            task: .imageEdit,
            prompt: "",
            focusedAssetId: "asset-1",
            referenceAssetIds: ["asset-1"]
        )
    )

    #expect(model.creationDraft.prompt == "reference.png")
}

@Test
@MainActor
func preferredLibraryWorkspaceIdPrefersSelectedThenActiveThenFirstVisibleWorkspace() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.importedAssets = [
        ImportedAssetRecord(
            id: "asset-1",
            workspaceId: "project-b",
            title: "first.png",
            sourcePath: "/tmp/first.png",
            storageKey: "imports/first.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        ),
        ImportedAssetRecord(
            id: "asset-2",
            workspaceId: "project-a",
            title: "second.png",
            sourcePath: "/tmp/second.png",
            storageKey: "imports/second.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        ),
    ]

    model.selectedLibraryWorkspaceId = "project-a"
    model.activeWorkspaceId = "project-b"
    #expect(model.preferredLibraryWorkspaceId == "project-a")

    model.selectedLibraryWorkspaceId = "missing-project"
    #expect(model.preferredLibraryWorkspaceId == "project-b")
}

@Test
@MainActor
func preferredLibraryWorkspaceIdFallsBackToStableDefaultWorkspaceForUnscopedAssets() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.workspaces = [
        WorkspaceRecord(id: "default-workspace", title: "Default"),
        WorkspaceRecord(id: "project-a", title: "Project A"),
    ]
    model.activeWorkspaceId = "project-a"
    model.selectedLibraryWorkspaceId = nil
    model.importedAssets = [
        ImportedAssetRecord(
            id: "asset-1",
            workspaceId: nil,
            title: "legacy.png",
            sourcePath: "/tmp/legacy.png",
            storageKey: "imports/legacy.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        )
    ]

    #expect(model.defaultWorkspaceId == "default-workspace")
    #expect(model.preferredLibraryWorkspaceId == "default-workspace")
    #expect(model.resolvedWorkspaceId(for: model.libraryAssets[0]) == "default-workspace")
}

@Test
@MainActor
func createWorkspaceWithoutSelectingKeepsCurrentWorkspace() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.activeWorkspaceId = "default-workspace"
    model.selectedLibraryWorkspaceId = nil

    let workspace = model.createWorkspace(select: false)

    #expect(model.activeWorkspaceId == "default-workspace")
    #expect(model.selectedLibraryWorkspaceId == nil)
    #expect(model.workspaces.contains(where: { $0.id == workspace.id }))
}

@Test
@MainActor
func createFreshWorkspaceSelectsWorkspaceAndStartsBlankDraft() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.activeWorkspaceId = "project-a"
    model.selectedLibraryWorkspaceId = "project-a"
    model.creationDraft = CreationDraft(
        workspaceId: "project-a",
        task: .videoGenerate,
        prompt: "Keep this old prompt",
        negativePrompt: "old negative",
        selectedModelId: "legacy-model",
        selectedAssetId: "asset-1",
        referenceAssetIds: ["asset-2"],
        lastActiveRunGroupId: "group-1",
        preferredDisplayedAssetId: "asset-1"
    )

    let workspace = model.createFreshWorkspace()

    #expect(model.activeWorkspaceId == workspace.id)
    #expect(model.selectedLibraryWorkspaceId == workspace.id)
    #expect(model.creationDraft.workspaceId == workspace.id)
    #expect(model.creationDraft.task == .videoGenerate)
    #expect(model.creationDraft.prompt.isEmpty)
    #expect(model.creationDraft.negativePrompt.isEmpty)
    #expect(model.creationDraft.selectedAssetId == nil)
    #expect(model.creationDraft.referenceAssetIds.isEmpty)
    #expect(model.creationDraft.lastActiveRunGroupId == nil)
    #expect(model.creationDraft.preferredDisplayedAssetId == nil)
}

@Test
@MainActor
func selectWorkspaceMaterializesDeferredWorkspaceId() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.workspaces = [WorkspaceRecord(id: "default-workspace", title: "New Project")]

    model.selectWorkspace("imported-project")

    #expect(model.activeWorkspaceId == "imported-project")
    #expect(model.selectedLibraryWorkspaceId == "imported-project")
    #expect(model.workspaces.contains(where: { $0.id == "imported-project" }))
    #expect(model.creationDraft.workspaceId == "imported-project")
}

@Test
@MainActor
func selectingNewWorkspaceClearsProjectScopedDraftContext() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.creationDraft.selectedAssetId = "asset-1"
    model.creationDraft.referenceAssetIds = ["asset-1", "asset-2"]
    model.creationDraft.lastActiveRunGroupId = "group-1"
    model.creationDraft.preferredDisplayedAssetId = "asset-2"

    _ = model.createWorkspace()

    #expect(model.creationDraft.selectedAssetId == nil)
    #expect(model.creationDraft.referenceAssetIds.isEmpty)
    #expect(model.creationDraft.lastActiveRunGroupId == nil)
    #expect(model.creationDraft.preferredDisplayedAssetId == nil)
}

@Test
@MainActor
func importingIntoDifferentWorkspaceClearsStaleProjectScopedDraftContext() async throws {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.workspaces = [
        WorkspaceRecord(id: "default-workspace", title: "New Project"),
        WorkspaceRecord(id: "project-a", title: "Project A"),
        WorkspaceRecord(id: "project-b", title: "Project B"),
    ]
    model.activeWorkspaceId = "project-a"
    model.selectedLibraryWorkspaceId = "project-a"
    model.creationDraft.workspaceId = "project-a"
    model.creationDraft.selectedAssetId = "asset-a"
    model.creationDraft.referenceAssetIds = ["asset-a", "asset-a-ref"]
    model.creationDraft.lastActiveRunGroupId = "group-a"
    model.creationDraft.preferredDisplayedAssetId = "asset-a"

    let sourceURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: sourceURL)
    defer { try? FileManager.default.removeItem(at: sourceURL) }

    let imported = await model.importExternalAssets(from: [sourceURL], workspaceId: "project-b")

    #expect(imported.count == 1)
    #expect(model.activeWorkspaceId == "project-b")
    #expect(model.selectedLibraryWorkspaceId == "project-b")
    #expect(model.creationDraft.workspaceId == "project-b")
    #expect(model.creationDraft.selectedAssetId == nil)
    #expect(model.creationDraft.referenceAssetIds.isEmpty)
    #expect(model.creationDraft.lastActiveRunGroupId == nil)
    #expect(model.creationDraft.preferredDisplayedAssetId == nil)

    if let importedAssetId = imported.first?.id {
        await model.removeImportedAsset(assetId: importedAssetId)
    }
}

@Test
@MainActor
func duplicateTopLevelImportDoesNotMaterializeEmptyWorkspace() async throws {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.workspaces = [
        WorkspaceRecord(id: "default-workspace", title: "New Project"),
        WorkspaceRecord(id: "project-a", title: "Project A"),
    ]
    model.activeWorkspaceId = "project-a"
    model.selectedLibraryWorkspaceId = "project-a"
    model.creationDraft.workspaceId = "project-a"

    let sourceURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: sourceURL)
    defer { try? FileManager.default.removeItem(at: sourceURL) }

    let firstImport = await model.importExternalAssets(from: [sourceURL], workspaceId: "new-project")
    #expect(firstImport.count == 1)
    #expect(model.activeWorkspaceId == "new-project")

    model.activeWorkspaceId = "project-a"
    model.selectedLibraryWorkspaceId = "project-a"
    model.creationDraft.workspaceId = "project-a"

    let duplicateImport = await model.importExternalAssets(from: [sourceURL], workspaceId: "ghost-project")

    #expect(duplicateImport.isEmpty)
    #expect(model.activeWorkspaceId == "project-a")
    #expect(model.selectedLibraryWorkspaceId == "project-a")
    #expect(model.creationDraft.workspaceId == "project-a")
    #expect(model.workspaces.contains(where: { $0.id == "ghost-project" }) == false)

    if let importedAssetId = firstImport.first?.id {
        await model.removeImportedAsset(assetId: importedAssetId)
    }
}

@Test
@MainActor
func runContextUsesDraftWorkspaceInsteadOfGlobalActiveWorkspace() {
    let model = MLXRAppModel(runtime: MockRuntime())
    model.activeWorkspaceId = "default-workspace"
    model.creationDraft.workspaceId = "project-2"

    let context = model.prepareRunContext(
        task: .imageGenerate,
        title: "Sunset skyline",
        sourceAssetIds: ["asset-1"]
    )

    #expect(context.workspaceId == "project-2")
    #expect(context.intentLabel == "Sunset skyline")
    #expect(context.sourceAssetIds == ["asset-1"])
}

@Test
@MainActor
func imageEditPlanningIntentKeepsSeededReferenceWithoutExistingPlan() async {
    let mock = MockRuntime()
    mock.supportedModels = [
        SupportedModelDescriptor(
            modelId: "flux2-klein-9b-local",
            displayName: "FLUX.2 Klein 9B",
            family: "flux2",
            familyVariant: nil,
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.edit"],
            provider: "huggingface",
            sourceSummary: "black-forest-labs/FLUX.2-Klein",
            license: nil,
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        )
    ]
    let model = MLXRAppModel(runtime: mock)
    await model.refresh()
    model.importedAssets = [
        ImportedAssetRecord(
            id: "asset-1",
            workspaceId: "default-workspace",
            title: "reference.png",
            sourcePath: "/tmp/reference.png",
            storageKey: "imports/reference.png",
            mediaType: "image/png",
            kind: .image,
            importedAt: Date()
        )
    ]

    model.seedComposer(
        with: ComposerSeedRequest(
            task: .imageEdit,
            prompt: "edit the source image",
            focusedAssetId: "asset-1",
            referenceAssetIds: ["asset-1"]
        )
    )

    let intent = model.creationPlanningIntent()

    #expect(intent?.task == ProductTask.imageEdit.rawValue)
    #expect(intent?.references.count == 1)
    #expect(intent?.references.first?.kind == .image)
}

@Test
@MainActor
func refreshReconstructsRunGroupsFromJobContext() async {
    let now = Date()
    let artifact = OutputArtifactRecord(
        artifactId: "art_1",
        artifactFormat: "png",
        role: "primary",
        exportable: true,
        metadata: [:],
        jobId: "job-run-group",
        filename: "result.png",
        mediaType: "image/png",
        sizeBytes: 1024,
        storageKey: "outputs/job-run-group/result.png",
        createdAt: now
    )
    let context = WorkflowContextMetadata(
        workspaceId: "default-workspace",
        runGroupId: "run-group-1",
        sourceAssetIds: ["source-asset-1"],
        intentLabel: "Make Image",
        presetId: "square"
    )

    let mock = MockRuntime()
    mock.jobs = [
        JobRecord(
            jobId: "job-run-group",
            request: JobRequest(
                modelId: "z-image-turbo-local",
                task: "image.generate",
                inputs: ["prompt": .string("golden retriever")],
                params: [:],
                output: .init(),
                context: context,
                extensions: [:]
            ),
            state: .completed,
            createdAt: now,
            updatedAt: now,
            error: nil,
            artifacts: [artifact]
        )
    ]

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.runGroups.count == 1)
    #expect(model.runGroups[0].id == "run-group-1")
    #expect(model.runGroups[0].title == "Make Image")
    #expect(model.runGroups[0].jobIds == ["job-run-group"])
    #expect(model.runGroups[0].assetIds == ["art_1"])
    #expect(model.runGroups[0].state == .completed)
}

@Test
@MainActor
func activityRunGroupsIncludeLegacyJobsWithoutContext() async {
    let now = Date()
    let mock = MockRuntime()
    mock.jobs = [
        JobRecord(
            jobId: "job_legacy",
            request: JobRequest(
                modelId: "z-image-turbo-local",
                task: "image.generate",
                inputs: ["prompt": .string("A lighthouse in a storm")],
                params: [:],
                output: .init(),
                context: nil,
                extensions: [:]
            ),
            state: .failed,
            createdAt: now,
            updatedAt: now,
            error: "boom",
            artifacts: []
        )
    ]

    let model = MLXRAppModel(runtime: mock)
    await model.refresh()

    #expect(model.activityRunGroups.count == 1)
    #expect(model.activityRunGroups[0].id == "legacy-job_legacy")
    #expect(model.activityRunGroups[0].title == "A lighthouse in a storm")
    #expect(model.activityRunGroups[0].state == .failed)
}
