import Foundation
import MLXRAppDomain

@MainActor
extension MLXRAppModel {
    public var composerPresentation: WorkflowPlanPresentation? {
        creationPlanResult?.presentation
    }

    public var composerAvailableModes: [TaskCategory] {
        let plannedModes = Set(
            composerSubworkflowOptions.compactMap { TaskCategory(rawValue: $0.mode) }
        )
        if plannedModes.isEmpty {
            return TaskCategory.allCases
        }
        return TaskCategory.allCases.filter { plannedModes.contains($0) }
    }

    public var composerSubworkflowOptions: [WorkflowPresentationSubworkflow] {
        composerPresentation?.subworkflows ?? []
    }

    public var composerMode: TaskCategory {
        TaskCategory(rawValue: composerPresentation?.primaryMode ?? creationDraft.task.category.rawValue)
            ?? creationDraft.task.category
    }

    public func composerSubworkflowOptions(for mode: TaskCategory) -> [WorkflowPresentationSubworkflow] {
        composerSubworkflowOptions.filter { $0.mode == mode.rawValue }
    }

    public var composerSelectedModelName: String? {
        selectedModel()?.displayName
    }

    public var composerFocusedAssetLabel: String? {
        guard let selectedAssetId = creationDraft.selectedAssetId,
              let asset = libraryAssets.first(where: { $0.id == selectedAssetId })
        else {
            return nil
        }
        return asset.displayTitle
    }

    public var composerReferenceSummaryLabel: String? {
        let referenceAssets = libraryAssets.filter { creationDraft.referenceAssetIds.contains($0.id) }
        guard !referenceAssets.isEmpty else {
            return nil
        }
        let total = referenceAssets.count
        let images = referenceAssets.filter(\.isImage).count
        let videos = referenceAssets.filter(\.isVideo).count
        let audio = referenceAssets.filter(\.isAudio).count

        var parts: [String] = []
        if images > 0 {
            parts.append("\(images) image")
        }
        if videos > 0 {
            parts.append("\(videos) video")
        }
        if audio > 0 {
            parts.append("\(audio) audio")
        }

        let detail = parts.joined(separator: ", ")
        if detail.isEmpty {
            return total == 1 ? "1 reference" : "\(total) references"
        }
        let noun = total == 1 ? "reference" : "references"
        return "\(detail) \(noun)"
    }

    public var composerRuntimeStatusLabel: String {
        if bootstrapError != nil {
            return "Runtime unavailable"
        }
        if runtimeStatus == nil {
            return "Boot"
        }
        if activeInstallCount > 0 {
            return "Installing model"
        }
        if isSubmittingImage || isSubmittingVideo {
            return "Generating"
        }
        if isPlanningCreation {
            return "Planning"
        }
        if selectedModel()?.installed != true {
            return "Model needed"
        }
        return "Ready"
    }

    public func selectComposerMode(_ mode: TaskCategory) {
        let options = composerSubworkflowOptions(for: mode)
        if let selected = options.first(where: { $0.task == creationDraft.task.rawValue }),
           ProductTask(rawValue: selected.task) != nil
        {
            return
        }
        if let targetTask = options.first(where: \.default)?.task ?? options.first?.task,
           let productTask = ProductTask(rawValue: targetTask)
        {
            selectComposerTask(productTask)
            return
        }
        let fallbackTask: ProductTask = switch mode {
        case .image: .imageGenerate
        case .video: .videoGenerate
        }
        selectComposerTask(fallbackTask)
    }

    public func selectComposerTask(_ task: ProductTask) {
        creationDraft.task = task
        syncWorkspaceDefaultsForTask()
    }

    public func selectComposerQuality(_ value: String) {
        guard let preset = QualityPreset(composerValue: value) else { return }
        creationDraft.qualityPreset = preset
        if !creationDraft.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
    }

    public func selectComposerAspect(_ value: String) {
        guard let preset = AspectPreset(composerValue: value) else { return }
        creationDraft.aspectPreset = preset
        if !creationDraft.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
    }

    public func selectComposerDuration(_ value: String) {
        guard let preset = DurationPreset(composerValue: value) else { return }
        creationDraft.durationPreset = preset
        if !creationDraft.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
    }

    public func selectComposerVariationCount(_ count: Int) {
        creationDraft.variationCount = max(1, count)
    }

    @discardableResult
    public func submitCurrentWorkspace(targetWorkspaceId: String? = nil) async -> String? {
        guard canSubmitCurrentWorkspace() else { return nil }
        guard let model = selectedModel(), model.installed else { return nil }
        let currentTask = creationDraft.task
        let settings = resolvedSettings()
        let references = await resolveComposerReferenceInputs()
        let sourceAssetIds = Array(
            Set(creationDraft.referenceAssetIds + [creationDraft.selectedAssetId].compactMap { $0 })
        )
        .sorted()
        let runGroupTitle = creationDraft.prompt.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        let effectiveWorkspaceId = targetWorkspaceId ?? creationDraft.workspaceId
        let context = prepareRunContext(
            task: currentTask,
            title: runGroupTitle,
            sourceAssetIds: sourceAssetIds,
            workspaceId: effectiveWorkspaceId
        )

        var acceptedAnyRun = false

        if currentTask.category == .image {
            let variationCount = max(1, creationDraft.variationCount)
            let baseSeed = Int(creationDraft.seed)
            for index in 0..<variationCount {
                let variationSeed = baseSeed.map { $0 + index }
                let accepted = await submitImage(
                    ImageGenerationRequest(
                        task: currentTask,
                        modelId: model.modelId,
                        prompt: creationDraft.prompt,
                        negativePrompt: creationDraft.negativePrompt,
                        width: settings.width,
                        height: settings.height,
                        numInferenceSteps: settings.numInferenceSteps,
                        guidanceScale: settings.guidanceScale,
                        seed: variationSeed,
                        artifactFormat: creationDraft.artifactFormat,
                        quality: creationDraft.qualityPreset.quality,
                        context: context,
                        runGroupTitle: runGroupTitle,
                        variationCount: variationCount,
                        references: references,
                        familyExtensions: selectedPack()?.familyExtensions ?? [:]
                    )
                )
                acceptedAnyRun = acceptedAnyRun || accepted
            }
        } else {
            acceptedAnyRun = await submitVideo(
                VideoGenerationRequest(
                    task: currentTask,
                    modelId: model.modelId,
                    prompt: creationDraft.prompt,
                    negativePrompt: creationDraft.negativePrompt,
                    width: settings.width,
                    height: settings.height,
                    numFrames: settings.numFrames,
                    fps: settings.fps,
                    numInferenceSteps: settings.numInferenceSteps,
                    guidanceScale: settings.guidanceScale,
                    seed: Int(creationDraft.seed),
                    artifactFormat: creationDraft.artifactFormat,
                    quality: creationDraft.qualityPreset.quality,
                    context: context,
                    runGroupTitle: runGroupTitle,
                    variationCount: 1,
                    references: references,
                    workflowVariant: selectedPack()?.workflowVariant,
                    controlVariant: selectedPack()?.controlVariant,
                    conditioningAttentionStrength: selectedPack()?.conditioningAttentionStrength,
                    familyExtensions: selectedPack()?.familyExtensions ?? [:]
                )
            )
        }

        return acceptedAnyRun ? context.workspaceId : nil
    }

    public func canSubmitCurrentWorkspace() -> Bool {
        selectedModel()?.installed == true
            && creationPlanError == nil
            && (creationPlanResult?.readiness.ready ?? false)
    }

    public func currentWorkspaceSubmitDisabledReason() -> String? {
        if selectedModel() == nil {
            return "Pick a model row that supports this workflow."
        }
        if selectedModel()?.installed != true {
            return "Install the selected model or switch to an installed row before running."
        }
        if isPlanningCreation {
            return "Checking the draft against the runtime plan…"
        }
        if let creationPlanError {
            return creationPlanError
        }
        return creationPlanResult?.readiness.blockingIssues.first
    }

    private func resolveComposerReferenceInputs() async -> [MediaReferenceInput] {
        var resolved: [MediaReferenceInput] = []
        for assetId in creationDraft.referenceAssetIds {
            guard
                let asset = libraryAssets.first(where: { $0.id == assetId }),
                let referenceKind = referenceKind(for: asset),
                let url = await resolvedURL(for: asset)
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
}

private extension QualityPreset {
    init?(composerValue: String) {
        switch composerValue {
        case "draft":
            self = .draft
        case "standard":
            self = .standard
        case "cinema":
            self = .cinematic
        default:
            return nil
        }
    }
}

private extension AspectPreset {
    init?(composerValue: String) {
        switch composerValue {
        case "square":
            self = .square
        case "landscape":
            self = .landscape
        case "portrait":
            self = .portrait
        case "story":
            self = .story
        default:
            return nil
        }
    }
}

private extension DurationPreset {
    init?(composerValue: String) {
        switch composerValue {
        case "short":
            self = .short
        case "medium":
            self = .medium
        case "long":
            self = .long
        default:
            return nil
        }
    }
}
