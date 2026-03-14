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
    private let isBusy: Bool
    private let error: String?
    private let onDismissError: () -> Void
    private let onResetDraft: () -> Void
    private let onSyncWorkspaceDefaults: () -> Void
    private let onPrepareRunContext: (_ task: ProductTask, _ title: String, _ sourceAssetIds: [String], _ variationCount: Int) -> WorkflowContextMetadata
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
        isBusy: Bool,
        error: String?,
        onDismissError: @escaping () -> Void,
        onResetDraft: @escaping () -> Void,
        onSyncWorkspaceDefaults: @escaping () -> Void,
        onPrepareRunContext: @escaping (_ task: ProductTask, _ title: String, _ sourceAssetIds: [String], _ variationCount: Int) -> WorkflowContextMetadata,
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
        self.isBusy = isBusy
        self.error = error
        self.onDismissError = onDismissError
        self.onResetDraft = onResetDraft
        self.onSyncWorkspaceDefaults = onSyncWorkspaceDefaults
        self.onPrepareRunContext = onPrepareRunContext
        self.onImportAssets = onImportAssets
        self.onResolveAssetURL = onResolveAssetURL
        self.onQueueInstall = onQueueInstall
        self.onSubmitImage = onSubmitImage
        self.onSubmitVideo = onSubmitVideo
    }

    public var body: some View {
        HSplitView {
            StudioSidebarView(
                workflows: workflows,
                selectedTask: Binding(
                    get: { workspace.task },
                    set: { newTask in
                        workspace.task = newTask
                    }
                ),
                recentAssets: Array(libraryAssets.prefix(18)),
                selectedAssetId: Binding(
                    get: { workspace.selectedAssetId },
                    set: { workspace.selectedAssetId = $0 }
                ),
                referenceAssetIds: Binding(
                    get: { Set(workspace.referenceAssetIds) },
                    set: { workspace.referenceAssetIds = Array($0).sorted() }
                ),
                onImport: { isPickingImports = true },
                onResetDraft: onResetDraft
            )
            .frame(minWidth: 260, idealWidth: 290, maxWidth: 340)

            StudioCanvasView(
                task: workspace.task,
                prompt: Binding(
                    get: { workspace.prompt },
                    set: { workspace.prompt = $0 }
                ),
                selectedAsset: displayedAsset,
                selectedAssetURL: displayedAssetURL,
                referenceAssets: referenceAssets,
                currentJob: currentJob,
                currentJobPhase: currentJobPhase,
                currentRunGroupTitle: currentRunGroup?.title,
                currentRunGroupAssets: currentRunGroupAssets,
                focusedResultAssetId: Binding(
                    get: { workspace.preferredDisplayedAssetId },
                    set: { workspace.preferredDisplayedAssetId = $0 }
                ),
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
                onSelectModel: { workspace.selectedModelId = $0 },
                onSelectPack: { workspace.selectedPackId = $0 },
                onQueueInstall: { modelId in
                    await onQueueInstall(modelId)
                },
                onResetToRecommended: {
                    workspace.useCustomSettings = false
                    onSyncWorkspaceDefaults()
                }
            )
            .frame(minWidth: 330, idealWidth: 380, maxWidth: 440)
        }
        .background(AdaptiveBackground())
        .animation(MLXRMotion.snappy, value: workspace.task)
        .animation(MLXRMotion.snappy, value: workspace.selectedAssetId)
        .animation(MLXRMotion.snappy, value: workspace.lastActiveRunGroupId)
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
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result else { return }
            Task {
                let imported = await onImportAssets(urls)
                if let first = imported.first {
                    workspace.selectedAssetId = first.id
                    if compatibleReferenceKind(for: libraryAssets.first(where: { $0.id == first.id }) ?? LibraryAsset.fromImported(imported).first!) != nil {
                        workspace.referenceAssetIds = [first.id]
                    }
                }
            }
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
        return jobs.first(where: { !$0.state.isTerminal })
    }

    private var currentJobPhase: String? {
        guard let currentJob else { return nil }
        return activePhases[currentJob.jobId]
    }

    private var resolvedSettings: StudioResolvedSettings {
        if workspace.useCustomSettings {
            return StudioResolvedSettings(
                width: Int(workspace.manualWidth),
                height: Int(workspace.manualHeight),
                numFrames: Int(workspace.manualFrames),
                fps: Int(workspace.manualFps),
                numInferenceSteps: Int(workspace.manualSteps),
                guidanceScale: workspace.manualGuidance
            )
        }

        let dimensions = workspace.aspectPreset.dimensions(for: workspace.task.category)
        let imageStepsByFamily: [String: (Int, Int, Int)] = [
            "flux2": (4, 8, 12),
            "qwen_image": (12, 24, 36),
            "z_image": (6, 12, 20),
        ]
        let defaultImageSteps = imageStepsByFamily[selectedModel?.family ?? ""] ?? (8, 16, 24)
        let imageSteps: Int = switch workspace.qualityPreset {
        case .draft: defaultImageSteps.0
        case .standard: defaultImageSteps.1
        case .cinematic: defaultImageSteps.2
        }
        let videoSteps: Int = switch workspace.qualityPreset {
        case .draft: 18
        case .standard: 28
        case .cinematic: 40
        }

        return StudioResolvedSettings(
            width: dimensions.width,
            height: dimensions.height,
            numFrames: workspace.task.category == .video ? workspace.durationPreset.numFrames : 1,
            fps: workspace.task.category == .video ? 24 : 1,
            numInferenceSteps: workspace.task.category == .image ? imageSteps : videoSteps,
            guidanceScale: workspace.task.category == .image ? 4.0 : 3.0
        )
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
        guard !workspace.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        guard let selectedModel else {
            return false
        }
        return selectedModel.installed
    }

    private var submitDisabledReason: String? {
        if workspace.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Add a prompt so MLXR knows what to make or change."
        }
        if selectedModel == nil {
            return "Pick a model row that supports this workflow."
        }
        if selectedModel?.installed == false {
            return "Install the selected model or switch to an installed row before running."
        }
        return nil
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
        if !workspace.referenceAssetIds.contains(assetId) {
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
            guard let model = selectedModel else { return }
            let references = await resolveReferenceInputs()
            let sourceAssetIds = Array(Set(workspace.referenceAssetIds + [workspace.selectedAssetId].compactMap { $0 })).sorted()
            let context = onPrepareRunContext(
                workspace.task,
                workspace.prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                sourceAssetIds,
                workspace.task.category == .image ? workspace.variationCount : 1
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

    private func compatibleReferenceKind(for asset: LibraryAsset) -> WorkflowReferenceKind? {
        switch workspace.task {
        case .imageGenerate, .videoGenerate:
            return nil
        case .imageEdit, .videoConditionImage, .videoInterpolate:
            return asset.isImage ? .image : nil
        case .videoConditionAudio:
            return asset.isAudio ? .audio : nil
        case .videoConditionVideo, .videoRetake:
            return asset.isVideo ? .video : nil
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
}
