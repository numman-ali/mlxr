// Transitional migration seam. Do not extend this screen for new product work;
// prefer the shell-mounted composer and project-first library flows.

import MLXRAppDomain
import MLXRDesignSystem
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
    private let error: String?
    private let onDismissError: () -> Void
    private let onDismissPlanningError: () -> Void
    private let onResetDraft: () -> Void
    private let onSyncWorkspaceDefaults: () -> Void
    private let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    private let onResolveAssetURL: @Sendable (LibraryAsset) async -> URL?
    private let onQueueInstall: @Sendable (String) async -> Void

    @State private var isPickingImports = false
    @State private var displayedAssetURL: URL?

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
        error: String?,
        onDismissError: @escaping () -> Void,
        onDismissPlanningError: @escaping () -> Void,
        onResetDraft: @escaping () -> Void,
        onSyncWorkspaceDefaults: @escaping () -> Void,
        onImportAssets: @escaping @Sendable ([URL]) async -> [ImportedAssetRecord],
        onResolveAssetURL: @escaping @Sendable (LibraryAsset) async -> URL?,
        onQueueInstall: @escaping @Sendable (String) async -> Void
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
        self.error = error
        self.onDismissError = onDismissError
        self.onDismissPlanningError = onDismissPlanningError
        self.onResetDraft = onResetDraft
        self.onSyncWorkspaceDefaults = onSyncWorkspaceDefaults
        self.onImportAssets = onImportAssets
        self.onResolveAssetURL = onResolveAssetURL
        self.onQueueInstall = onQueueInstall
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
            showsWorkflowCard: false,
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
            error: error,
            onDismissError: onDismissError,
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
}
