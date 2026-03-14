import Foundation
import MLXRAppDomain

@MainActor
extension MLXRAppModel {
    public var composerPresentation: WorkflowPlanPresentation? {
        studioPlanResult?.presentation
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
        TaskCategory(rawValue: composerPresentation?.primaryMode ?? studioWorkspace.task.category.rawValue)
            ?? studioWorkspace.task.category
    }

    public func composerSubworkflowOptions(for mode: TaskCategory) -> [WorkflowPresentationSubworkflow] {
        composerSubworkflowOptions.filter { $0.mode == mode.rawValue }
    }

    public var composerSelectedModelName: String? {
        selectedModel()?.displayName
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
        if isPlanningStudio {
            return "Planning"
        }
        if selectedModel()?.installed != true {
            return "Model needed"
        }
        return "Ready"
    }

    public func selectComposerMode(_ mode: TaskCategory) {
        let options = composerSubworkflowOptions(for: mode)
        if let selected = options.first(where: { $0.task == studioWorkspace.task.rawValue }),
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
        studioWorkspace.task = task
        syncWorkspaceDefaultsForTask()
        scheduleStudioPlan()
    }

    public func selectComposerQuality(_ value: String) {
        guard let preset = QualityPreset(composerValue: value) else { return }
        studioWorkspace.qualityPreset = preset
        if !studioWorkspace.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
        scheduleStudioPlan()
    }

    public func selectComposerAspect(_ value: String) {
        guard let preset = AspectPreset(composerValue: value) else { return }
        studioWorkspace.aspectPreset = preset
        if !studioWorkspace.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
        scheduleStudioPlan()
    }

    public func selectComposerDuration(_ value: String) {
        guard let preset = DurationPreset(composerValue: value) else { return }
        studioWorkspace.durationPreset = preset
        if !studioWorkspace.useCustomSettings {
            syncWorkspaceDefaultsForTask()
        }
        scheduleStudioPlan()
    }

    public func selectComposerVariationCount(_ count: Int) {
        studioWorkspace.variationCount = max(1, count)
        scheduleStudioPlan()
    }

    public func submitCurrentWorkspace() async {
        guard canSubmitCurrentWorkspace() else { return }
        guard let model = selectedModel(), model.installed else { return }
        let currentTask = studioWorkspace.task
        let settings = resolvedSettings()
        let references = await resolveComposerReferenceInputs()
        let sourceAssetIds = Array(
            Set(studioWorkspace.referenceAssetIds + [studioWorkspace.selectedAssetId].compactMap { $0 })
        )
        .sorted()
        let runGroupTitle = studioWorkspace.prompt.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        let context = prepareRunContext(
            task: currentTask,
            title: runGroupTitle,
            sourceAssetIds: sourceAssetIds
        )

        if currentTask.category == .image {
            let variationCount = max(1, studioWorkspace.variationCount)
            let baseSeed = Int(studioWorkspace.seed)
            for index in 0..<variationCount {
                let variationSeed = baseSeed.map { $0 + index }
                await submitImage(
                    ImageGenerationRequest(
                        task: currentTask,
                        modelId: model.modelId,
                        prompt: studioWorkspace.prompt,
                        negativePrompt: studioWorkspace.negativePrompt,
                        width: settings.width,
                        height: settings.height,
                        numInferenceSteps: settings.numInferenceSteps,
                        guidanceScale: settings.guidanceScale,
                        seed: variationSeed,
                        artifactFormat: studioWorkspace.artifactFormat,
                        quality: studioWorkspace.qualityPreset.quality,
                        context: context,
                        runGroupTitle: runGroupTitle,
                        variationCount: variationCount,
                        references: references,
                        familyExtensions: selectedPack()?.familyExtensions ?? [:]
                    )
                )
            }
        } else {
            await submitVideo(
                VideoGenerationRequest(
                    task: currentTask,
                    modelId: model.modelId,
                    prompt: studioWorkspace.prompt,
                    negativePrompt: studioWorkspace.negativePrompt,
                    width: settings.width,
                    height: settings.height,
                    numFrames: settings.numFrames,
                    fps: settings.fps,
                    numInferenceSteps: settings.numInferenceSteps,
                    guidanceScale: settings.guidanceScale,
                    seed: Int(studioWorkspace.seed),
                    artifactFormat: studioWorkspace.artifactFormat,
                    quality: studioWorkspace.qualityPreset.quality,
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
    }

    public func canSubmitCurrentWorkspace() -> Bool {
        selectedModel()?.installed == true
            && studioPlanError == nil
            && (studioPlanResult?.readiness.ready ?? false)
    }

    public func currentWorkspaceSubmitDisabledReason() -> String? {
        if selectedModel() == nil {
            return "Pick a model row that supports this workflow."
        }
        if selectedModel()?.installed != true {
            return "Install the selected model or switch to an installed row before running."
        }
        if isPlanningStudio {
            return "Checking the draft against the runtime plan…"
        }
        if let studioPlanError {
            return studioPlanError
        }
        return studioPlanResult?.readiness.blockingIssues.first
    }

    private func resolveComposerReferenceInputs() async -> [MediaReferenceInput] {
        var resolved: [MediaReferenceInput] = []
        for assetId in studioWorkspace.referenceAssetIds {
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
