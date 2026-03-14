import MLXRAppDomain
import MLXRDesignSystem
import MLXRRecipes
import SwiftUI
import UniformTypeIdentifiers

@MainActor
public struct StudioScreen: View {
    @Binding private var workspace: StudioWorkspaceDraft

    private let catalog: CatalogSnapshot
    private let libraryAssets: [LibraryAsset]
    private let installOperations: [ModelInstallOperationRecord]
    private let jobs: [JobRecord]
    private let activePhases: [String: String]
    private let currentRunGroup: RunGroupRecord?
    private let currentRunGroupAssets: [LibraryAsset]
    private let packs: [PackRecord]
    private let planningResult: WorkflowPlanResult?
    private let planningError: String?
    private let isPlanning: Bool
    private let resolvedSettings: StudioResolvedSettings
    private let isBusy: Bool
    private let error: String?
    private let onDismissError: () -> Void
    private let onDismissPlanningError: () -> Void
    private let onResetDraft: () -> Void
    private let onSyncWorkspaceDefaults: () -> Void
    private let onPlanWorkspace: () -> Void
    private let onPrepareRunContext: (_ task: ProductTask, _ title: String, _ sourceAssetIds: [String]) -> WorkflowContextMetadata
    private let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    private let onResolveAssetURL: @Sendable (LibraryAsset) async -> URL?
    private let onQueueInstall: @Sendable (String) async -> Void
    private let onSubmitImage: @Sendable (ImageGenerationRequest) async -> Void
    private let onSubmitVideo: @Sendable (VideoGenerationRequest) async -> Void

    @State private var isPickingImports = false
    @State private var displayedAssetURL: URL?

    private let recipeCatalog = RecipeCatalog()

    public init(
        workspace: Binding<StudioWorkspaceDraft>,
        catalog: CatalogSnapshot,
        libraryAssets: [LibraryAsset],
        installOperations: [ModelInstallOperationRecord],
        jobs: [JobRecord],
        activePhases: [String: String],
        currentRunGroup: RunGroupRecord?,
        currentRunGroupAssets: [LibraryAsset],
        packs: [PackRecord],
        planningResult: WorkflowPlanResult?,
        planningError: String?,
        isPlanning: Bool,
        resolvedSettings: StudioResolvedSettings,
        isBusy: Bool,
        error: String?,
        onDismissError: @escaping () -> Void,
        onDismissPlanningError: @escaping () -> Void,
        onResetDraft: @escaping () -> Void,
        onSyncWorkspaceDefaults: @escaping () -> Void,
        onPlanWorkspace: @escaping () -> Void,
        onPrepareRunContext: @escaping (_ task: ProductTask, _ title: String, _ sourceAssetIds: [String]) -> WorkflowContextMetadata,
        onImportAssets: @escaping @Sendable ([URL]) async -> [ImportedAssetRecord],
        onResolveAssetURL: @escaping @Sendable (LibraryAsset) async -> URL?,
        onQueueInstall: @escaping @Sendable (String) async -> Void,
        onSubmitImage: @escaping @Sendable (ImageGenerationRequest) async -> Void,
        onSubmitVideo: @escaping @Sendable (VideoGenerationRequest) async -> Void
    ) {
        self._workspace = workspace
        self.catalog = catalog
        self.libraryAssets = libraryAssets
        self.installOperations = installOperations
        self.jobs = jobs
        self.activePhases = activePhases
        self.currentRunGroup = currentRunGroup
        self.currentRunGroupAssets = currentRunGroupAssets
        self.packs = packs
        self.planningResult = planningResult
        self.planningError = planningError
        self.isPlanning = isPlanning
        self.resolvedSettings = resolvedSettings
        self.isBusy = isBusy
        self.error = error
        self.onDismissError = onDismissError
        self.onDismissPlanningError = onDismissPlanningError
        self.onResetDraft = onResetDraft
        self.onSyncWorkspaceDefaults = onSyncWorkspaceDefaults
        self.onPlanWorkspace = onPlanWorkspace
        self.onPrepareRunContext = onPrepareRunContext
        self.onImportAssets = onImportAssets
        self.onResolveAssetURL = onResolveAssetURL
        self.onQueueInstall = onQueueInstall
        self.onSubmitImage = onSubmitImage
        self.onSubmitVideo = onSubmitVideo
    }

    public var body: some View {
        applyLifecycle(
            to: HSplitView {
                sidebarPane
                canvasPane
                inspectorPane
            }
            .background(AdaptiveBackground())
            .animation(MLXRMotion.snappy, value: workspace.task)
            .animation(MLXRMotion.snappy, value: workspace.selectedAssetId)
            .animation(MLXRMotion.snappy, value: workspace.lastActiveRunGroupId)
        )
    }

    private var sidebarPane: some View {
        StudioSidebarView(
            workflows: workflows,
            allowedReferenceKinds: allowedReferenceKinds,
            selectedTask: taskBinding,
            recentAssets: Array(libraryAssets.prefix(18)),
            selectedAssetId: selectedAssetIdBinding,
            referenceAssetIds: referenceAssetIdsBinding,
            onImport: { isPickingImports = true },
            onResetDraft: onResetDraft
        )
        .frame(minWidth: 260, idealWidth: 290, maxWidth: 340)
    }

    private var canvasPane: some View {
        StudioCanvasView(
            task: workspace.task,
            prompt: promptBinding,
            selectedAsset: displayedAsset,
            selectedAssetURL: displayedAssetURL,
            referenceAssets: referenceAssets,
            currentJob: currentJob,
            currentJobPhase: currentJobPhase,
            currentRunGroupTitle: currentRunGroup?.title,
            currentRunGroupAssets: currentRunGroupAssets,
            focusedResultAssetId: focusedResultAssetIdBinding,
            planningError: planningError,
            onDismissPlanningError: onDismissPlanningError,
            isBusy: isBusy,
            error: error,
            onDismissError: onDismissError,
            suggestions: selectedRecipe?.examplePrompts ?? [],
            canSubmit: canSubmit,
            submitDisabledReason: submitDisabledReason,
            onSubmit: submit,
            onUseFocusedAssetAsReference: addFocusedAssetAsReference,
            onEditAsset: { asset in seedWorkspace(from: asset, task: .imageEdit) },
            onAnimateAsset: { asset in seedWorkspace(from: asset, task: .videoConditionImage) }
        )
        .frame(minWidth: 620, idealWidth: 780, maxWidth: .infinity)
    }

    private var inspectorPane: some View {
        StudioInspectorView(
            workspace: $workspace,
            selectedModel: selectedModel,
            availableModels: availableModels,
            availablePacks: availablePacks,
            selectedPack: selectedPack,
            installOperation: selectedModel.flatMap { installOperation(for: $0.modelId) },
            resolvedSettings: resolvedSettings,
            shouldShowResetToRecommended: shouldShowResetToRecommended,
            referenceAssets: referenceAssets,
            referenceRequirements: planningResult?.readiness.referenceRequirements ?? [],
            readinessWarnings: planningResult?.readiness.warnings ?? [],
            isPlanning: isPlanning,
            onSelectModel: { workspace.selectedModelId = $0 },
            onSelectPack: { workspace.selectedPackId = $0 },
            onQueueInstall: onQueueInstall,
            onResetToRecommended: {
                workspace.useCustomSettings = false
                onSyncWorkspaceDefaults()
            }
        )
        .frame(minWidth: 330, idealWidth: 380, maxWidth: 440)
    }

    private var taskBinding: Binding<ProductTask> {
        Binding(
            get: { workspace.task },
            set: { workspace.task = $0 }
        )
    }

    private var promptBinding: Binding<String> {
        Binding(
            get: { workspace.prompt },
            set: { workspace.prompt = $0 }
        )
    }

    private var selectedAssetIdBinding: Binding<String?> {
        Binding(
            get: { workspace.selectedAssetId },
            set: { workspace.selectedAssetId = $0 }
        )
    }

    private var referenceAssetIdsBinding: Binding<Set<String>> {
        Binding(
            get: { Set(workspace.referenceAssetIds) },
            set: { workspace.referenceAssetIds = Array($0).sorted() }
        )
    }

    private var focusedResultAssetIdBinding: Binding<String?> {
        Binding(
            get: { workspace.preferredDisplayedAssetId },
            set: { workspace.preferredDisplayedAssetId = $0 }
        )
    }

    private var allowedReferenceKinds: Set<WorkflowReferenceKind> {
        Set((planningResult?.readiness.referenceRequirements ?? []).map(\.kind))
    }

    private func applyLifecycle<Content: View>(to content: Content) -> some View {
        content
            .task {
                onSyncWorkspaceDefaults()
                syncPreferredDisplayedAsset()
            }
            .onChange(of: workspace.task) { _, _ in
                workspace.selectedPackId = selectedPack?.id
                onSyncWorkspaceDefaults()
            }
            .onChange(of: workspace.selectedModelId) { _, _ in
                if let pack = selectedPack, pack.family != selectedModel?.family {
                    workspace.selectedPackId = nil
                }
                onSyncWorkspaceDefaults()
            }
            .onChange(of: workspace.qualityPreset) { _, _ in
                if !workspace.useCustomSettings {
                    onSyncWorkspaceDefaults()
                }
            }
            .onChange(of: workspace.aspectPreset) { _, _ in
                if !workspace.useCustomSettings {
                    onSyncWorkspaceDefaults()
                }
            }
            .onChange(of: workspace.durationPreset) { _, _ in
                if !workspace.useCustomSettings {
                    onSyncWorkspaceDefaults()
                }
            }
            .onChange(of: currentRunGroupAssets.map(\.id)) { _, _ in
                syncPreferredDisplayedAsset()
            }
            .task(id: planningKey) {
                onPlanWorkspace()
            }
            .task(id: displayedAsset?.id) {
                guard let displayedAsset else {
                    displayedAssetURL = nil
                    return
                }
                displayedAssetURL = await onResolveAssetURL(displayedAsset)
            }
            .fileImporter(
                isPresented: $isPickingImports,
                allowedContentTypes: [.image, .movie, .audio],
                allowsMultipleSelection: true,
                onCompletion: handleImportResult
            )
    }

    private func handleImportResult(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result else {
            return
        }
        Task {
            let imported = await onImportAssets(urls)
            handleImportedAssets(imported)
        }
    }

    private func handleImportedAssets(_ imported: [ImportedAssetRecord]) {
        guard let first = imported.first else {
            return
        }
        let importedAssets = LibraryAsset.fromImported(imported)
        guard let firstAsset = importedAssets.first else {
            return
        }
        workspace.selectedAssetId = first.id
        if compatibleReferenceKind(for: firstAsset) != nil {
            workspace.referenceAssetIds = [first.id]
        }
    }

    private var availableModels: [ModelCatalogItem] {
        catalog.items(for: workspace.task)
    }

    private var selectedModel: ModelCatalogItem? {
        availableModels.first(where: { $0.modelId == workspace.selectedModelId })
            ?? availableModels.first
    }

    private var availablePacks: [PackRecord] {
        guard let family = selectedModel?.family else { return [] }
        return packs.filter { $0.family == family && $0.supports(task: workspace.task) }
    }

    private var selectedPack: PackRecord? {
        if let selectedPackId = workspace.selectedPackId {
            return availablePacks.first(where: { $0.id == selectedPackId })
        }
        return availablePacks.first
    }

    private var selectedRecipe: Recipe? {
        recipeCatalog.defaultRecipe(for: workspace.task)
    }

    private var selectedAsset: LibraryAsset? {
        guard let selectedAssetId = workspace.selectedAssetId else { return nil }
        return libraryAssets.first(where: { $0.id == selectedAssetId })
    }

    private var referenceAssets: [LibraryAsset] {
        libraryAssets.filter {
            workspace.referenceAssetIds.contains($0.id) && assetSupportsCurrentTask($0)
        }
    }

    private var displayedAsset: LibraryAsset? {
        if let preferredDisplayedAssetId = workspace.preferredDisplayedAssetId,
           let preferred = libraryAssets.first(where: { $0.id == preferredDisplayedAssetId }) {
            return preferred
        }
        if let latestResult = currentRunGroupAssets.last {
            return latestResult
        }
        return selectedAsset
    }

    private var currentJob: JobRecord? {
        if let currentRunGroup {
            let groupedJobs = jobs.filter { currentRunGroup.jobIds.contains($0.jobId) }
            if let active = groupedJobs.first(where: { !$0.state.isTerminal }) {
                return active
            }
        }
        return nil
    }

    private var currentJobPhase: String? {
        guard let currentJob else { return nil }
        return activePhases[currentJob.jobId]
    }

    private var shouldShowResetToRecommended: Bool {
        guard workspace.useCustomSettings else { return false }
        return StudioResolvedSettings(
            width: Int(workspace.manualWidth),
            height: Int(workspace.manualHeight),
            numFrames: Int(workspace.manualFrames),
            fps: Int(workspace.manualFps),
            numInferenceSteps: Int(workspace.manualSteps),
            guidanceScale: workspace.manualGuidance
        ) != resolvedSettings
    }

    private var canSubmit: Bool {
        selectedModel?.installed == true
            && planningError == nil
            && (planningResult?.readiness.ready ?? false)
    }

    private var submitDisabledReason: String? {
        if selectedModel == nil {
            return "Pick a model row that supports this workflow."
        }
        if selectedModel?.installed != true {
            return "Install the selected model or switch to an installed row before running."
        }
        if isPlanning {
            return "Checking the draft against the runtime plan…"
        }
        if let planningError {
            return planningError
        }
        return planningResult?.readiness.blockingIssues.first
    }

    private var workflows: [StudioWorkflowOption] {
        [
            StudioWorkflowOption(id: "make-image", title: "Make Image", subtitle: "Start from a prompt", icon: "photo.fill", availability: .available(.imageGenerate)),
            StudioWorkflowOption(id: "edit-image", title: "Edit Image", subtitle: "Change an existing still", icon: "wand.and.stars", availability: .available(.imageEdit)),
            StudioWorkflowOption(id: "make-video", title: "Make Video", subtitle: "Create a clip from text", icon: "film.fill", availability: .available(.videoGenerate)),
            StudioWorkflowOption(id: "animate-image", title: "Animate Image", subtitle: "Turn a still into motion", icon: "sparkles.tv.fill", availability: .available(.videoConditionImage)),
            StudioWorkflowOption(id: "video-sound", title: "Video From Sound", subtitle: "Drive a clip from audio", icon: "waveform.and.mic", availability: .available(.videoConditionAudio)),
            StudioWorkflowOption(id: "guide-video", title: "Guide With Video", subtitle: "Use an existing clip as the guide", icon: "film.stack.fill", availability: .available(.videoConditionVideo)),
            StudioWorkflowOption(id: "blend-frames", title: "Blend Frames", subtitle: "Interpolate between key images", icon: "square.stack.3d.forward.dottedline", availability: .available(.videoInterpolate)),
            StudioWorkflowOption(id: "retake-clip", title: "Retake Clip", subtitle: "Redo part of a shot", icon: "scissors", availability: .available(.videoRetake)),
            StudioWorkflowOption(id: "upscale", title: "Upscale", subtitle: "Make an asset cleaner and larger", icon: "arrow.up.right.square", availability: .comingSoon),
            StudioWorkflowOption(id: "transcribe", title: "Transcribe", subtitle: "Turn audio into text", icon: "captions.bubble.fill", availability: .comingSoon),
            StudioWorkflowOption(id: "podcast", title: "Podcast Episode", subtitle: "Build spoken audio from a script", icon: "mic.circle.fill", availability: .comingSoon),
        ]
    }

    private func installOperation(for modelId: String) -> ModelInstallOperationRecord? {
        installOperations.first { $0.modelId == modelId && !$0.phase.isTerminal }
            ?? installOperations.first { $0.modelId == modelId }
    }

    private func addFocusedAssetAsReference() {
        guard let assetId = displayedAsset?.id else { return }
        if compatibleReferenceKind(for: displayedAsset) != nil && !workspace.referenceAssetIds.contains(assetId) {
            workspace.referenceAssetIds.append(assetId)
        }
    }

    private func syncPreferredDisplayedAsset() {
        if let preferredDisplayedAssetId = workspace.preferredDisplayedAssetId,
           libraryAssets.contains(where: { $0.id == preferredDisplayedAssetId }) {
            return
        }
        workspace.preferredDisplayedAssetId = currentRunGroupAssets.last?.id ?? workspace.selectedAssetId
    }

    private func submit() {
        Task {
            guard canSubmit else { return }
            guard let model = selectedModel else { return }
            let references = await resolveReferenceInputs()
            let sourceAssetIds = Array(Set(workspace.referenceAssetIds + [workspace.selectedAssetId].compactMap { $0 })).sorted()
            let runGroupTitle = workspace.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let context = onPrepareRunContext(
                workspace.task,
                runGroupTitle,
                sourceAssetIds
            )

            if workspace.task.category == .image {
                let variationCount = max(1, workspace.variationCount)
                let baseSeed = Int(workspace.seed)
                for index in 0..<variationCount {
                    let variationSeed = baseSeed.map { $0 + index }
                    await onSubmitImage(
                        ImageGenerationRequest(
                            task: workspace.task,
                            modelId: model.modelId,
                            prompt: workspace.prompt,
                            negativePrompt: workspace.negativePrompt,
                            width: resolvedSettings.width,
                            height: resolvedSettings.height,
                            numInferenceSteps: resolvedSettings.numInferenceSteps,
                            guidanceScale: resolvedSettings.guidanceScale,
                            seed: variationSeed,
                            artifactFormat: workspace.artifactFormat,
                            quality: workspace.qualityPreset.quality,
                            context: context,
                            runGroupTitle: runGroupTitle,
                            variationCount: variationCount,
                            references: references,
                            familyExtensions: selectedPack?.familyExtensions ?? [:]
                        )
                    )
                }
            } else {
                await onSubmitVideo(
                    VideoGenerationRequest(
                        task: workspace.task,
                        modelId: model.modelId,
                        prompt: workspace.prompt,
                        negativePrompt: workspace.negativePrompt,
                        width: resolvedSettings.width,
                        height: resolvedSettings.height,
                        numFrames: resolvedSettings.numFrames,
                        fps: resolvedSettings.fps,
                        numInferenceSteps: resolvedSettings.numInferenceSteps,
                        guidanceScale: resolvedSettings.guidanceScale,
                        seed: Int(workspace.seed),
                        artifactFormat: workspace.artifactFormat,
                        quality: workspace.qualityPreset.quality,
                        context: context,
                        runGroupTitle: runGroupTitle,
                        variationCount: 1,
                        references: references,
                        workflowVariant: selectedPack?.workflowVariant,
                        controlVariant: selectedPack?.controlVariant,
                        conditioningAttentionStrength: selectedPack?.conditioningAttentionStrength,
                        familyExtensions: selectedPack?.familyExtensions ?? [:]
                    )
                )
            }
        }
    }

    private func resolveReferenceInputs() async -> [MediaReferenceInput] {
        var resolved: [MediaReferenceInput] = []
        for asset in referenceAssets {
            guard
                let referenceKind = compatibleReferenceKind(for: asset),
                let url = await onResolveAssetURL(asset)
            else {
                continue
            }
            resolved.append(
                MediaReferenceInput(
                    fileURL: url,
                    kind: referenceKind,
                    role: "reference"
                )
            )
        }
        return resolved
    }

    private func assetSupportsCurrentTask(_ asset: LibraryAsset) -> Bool {
        compatibleReferenceKind(for: asset) != nil
    }

    private func compatibleReferenceKind(for asset: LibraryAsset?) -> WorkflowReferenceKind? {
        guard let asset else { return nil }
        let allowedKinds = Set((planningResult?.readiness.referenceRequirements ?? []).map(\.kind))
        guard !allowedKinds.isEmpty else { return nil }
        return switch asset.kind {
        case .image where allowedKinds.contains(.image):
            .image
        case .video where allowedKinds.contains(.video):
            .video
        case .audio where allowedKinds.contains(.audio):
            .audio
        default:
            nil
        }
    }

    private func seedWorkspace(from asset: LibraryAsset, task: ProductTask) {
        workspace.task = task
        workspace.selectedAssetId = asset.id
        workspace.preferredDisplayedAssetId = asset.id
        if !asset.prompt.isEmpty {
            workspace.prompt = asset.prompt
        }
        if compatibleReferenceKind(for: asset) != nil {
            workspace.referenceAssetIds = [asset.id]
        }
        onSyncWorkspaceDefaults()
    }

    private var planningKey: String {
        [
            workspace.task.rawValue,
            workspace.prompt,
            workspace.negativePrompt,
            workspace.selectedModelId,
            workspace.selectedPackId ?? "",
            workspace.qualityPreset.rawValue,
            workspace.aspectPreset.rawValue,
            workspace.durationPreset.rawValue,
            workspace.artifactFormat,
            workspace.useCustomSettings.description,
            workspace.manualWidth.description,
            workspace.manualHeight.description,
            workspace.manualFrames.description,
            workspace.manualFps.description,
            workspace.manualSteps.description,
            workspace.manualGuidance.description,
            workspace.referenceAssetIds.sorted().joined(separator: ","),
        ]
        .joined(separator: "|")
    }
}
