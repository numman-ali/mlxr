import Foundation
import MLXRAppDomain

@MainActor
extension MLXRAppModel {
    public var packCatalog: [PackRecord] {
        [
            PackRecord(
                id: "qwen-base",
                title: "Base",
                summary: "Balanced Qwen image defaults for polished stills.",
                category: .looks,
                family: "qwen_image",
                tasks: [.imageGenerate, .imageEdit]
            ),
            PackRecord(
                id: "qwen-lightning",
                title: "Lightning",
                summary: "Sharper, faster Qwen image timing for quick drafts.",
                category: .looks,
                family: "qwen_image",
                tasks: [.imageGenerate, .imageEdit],
                familyExtensions: ["scheduler_preset": .string("lightning")]
            ),
            PackRecord(
                id: "qwen-wuli",
                title: "Wuli Turbo",
                summary: "Aggressive fast-lane look for stylized Qwen image runs.",
                category: .looks,
                family: "qwen_image",
                tasks: [.imageGenerate, .imageEdit],
                familyExtensions: ["scheduler_preset": .string("turbo_wuli")]
            ),
            PackRecord(
                id: "ltx-standard-motion",
                title: "Standard Motion",
                summary: "The default LTX motion path for text, image, and audio driven clips.",
                category: .motion,
                family: "ltx",
                tasks: [.videoGenerate, .videoConditionImage, .videoConditionAudio]
            ),
            PackRecord(
                id: "ltx-motion-track",
                title: "Motion Track",
                summary: "Guided motion tracking for video control flows.",
                category: .motion,
                family: "ltx",
                tasks: [.videoConditionVideo],
                controlVariant: "motion_track_control"
            ),
            PackRecord(
                id: "ltx-video-guide",
                title: "Video Guide",
                summary: "Use a source clip to steer motion and timing.",
                category: .control,
                family: "ltx",
                tasks: [.videoConditionVideo],
                controlVariant: "ic_lora"
            ),
            PackRecord(
                id: "ltx-retake",
                title: "Retake",
                summary: "Optimized defaults for reworking a section of an existing clip.",
                category: .control,
                family: "ltx",
                tasks: [.videoRetake]
            ),
        ]
    }

    public var currentRunGroup: RunGroupRecord? {
        guard let runGroupId = studioWorkspace.lastActiveRunGroupId else { return nil }
        return runGroups.first(where: { $0.id == runGroupId })
    }

    public var currentRunGroupAssets: [LibraryAsset] {
        guard let runGroupId = studioWorkspace.lastActiveRunGroupId else { return [] }
        return libraryAssets.filter { $0.runGroupId == runGroupId }
            .sorted { $0.createdAt < $1.createdAt }
    }

    public var recentRunGroups: [RunGroupRecord] {
        runGroups.sorted { $0.updatedAt > $1.updatedAt }
    }

    public var activityRunGroups: [RunGroupRecord] {
        let retentionCutoff = Date().addingTimeInterval(-(48 * 60 * 60))
        let persistedIds = Set(runGroups.map(\.id))
        let legacyGroups = jobs.compactMap { job -> RunGroupRecord? in
            guard job.request.context?.runGroupId == nil else {
                return nil
            }
            guard let task = ProductTask.from(rawTask: job.request.task) else {
                return nil
            }
            let syntheticId = "legacy-\(job.jobId)"
            guard !persistedIds.contains(syntheticId) else {
                return nil
            }
            let relatedAssets = libraryAssets.filter { $0.jobId == job.jobId }
            let group = RunGroupRecord(
                id: syntheticId,
                workspaceId: activeWorkspaceId,
                collectionId: nil,
                task: task,
                title: legacyActivityTitle(for: job, task: task),
                sourceAssetIds: [],
                jobIds: [job.jobId],
                assetIds: relatedAssets.map(\.id),
                variationCount: max(relatedAssets.count, 1),
                createdAt: job.createdAt,
                updatedAt: job.updatedAt,
                state: state(for: [job], assetCount: relatedAssets.count)
            )
            return shouldIncludeInActivity(group, retentionCutoff: retentionCutoff)
                ? group
                : nil
        }
        let persistedGroups = runGroups.filter {
            shouldIncludeInActivity($0, retentionCutoff: retentionCutoff)
        }
        return (persistedGroups + legacyGroups).sorted { $0.updatedAt > $1.updatedAt }
    }

    public func dismissActivityRunGroup(_ runGroupId: String) {
        dismissedActivityRunGroupIds.insert(runGroupId)
    }

    public func persistPresentationState() {
        do {
            try workspaceStateStore.persist(
                AppPresentationState(
                    hasCompletedModelSetup: hasCompletedOnboarding,
                    activeWorkspaceId: activeWorkspaceId,
                    selectedLibraryWorkspaceId: selectedLibraryWorkspaceId,
                    workspaces: workspaces,
                    collections: collections,
                    runGroups: runGroups,
                    dismissedActivityRunGroupIds: Array(dismissedActivityRunGroupIds).sorted(),
                    assets: Array(assetRecords.values).sorted { $0.id < $1.id },
                    workspaceDraft: studioWorkspace
                )
            )
        } catch {
            settingsError = error.localizedDescription
        }
    }

    public func seedComposer(with request: ComposerSeedRequest) {
        ensureWorkspaceExists()
        if let workspaceId = request.workspaceId, workspaces.contains(where: { $0.id == workspaceId }) {
            selectWorkspace(workspaceId)
        }
        studioWorkspace.workspaceId = activeWorkspaceId
        studioWorkspace.task = request.task
        if let resolvedPrompt = request.prompt ?? request.focusedAssetId.flatMap({ assetId in
            libraryAssets.first(where: { $0.id == assetId })?.prompt
        }), !resolvedPrompt.isEmpty {
            studioWorkspace.prompt = resolvedPrompt
        }
        if let focusedAssetId = request.focusedAssetId {
            studioWorkspace.selectedAssetId = focusedAssetId
            studioWorkspace.preferredDisplayedAssetId = focusedAssetId
            markAssetUsed(focusedAssetId)
        } else {
            studioWorkspace.selectedAssetId = nil
            studioWorkspace.preferredDisplayedAssetId = nil
        }
        studioWorkspace.referenceAssetIds = request.referenceAssetIds
        if !request.referenceAssetIds.isEmpty {
            request.referenceAssetIds.forEach(markAssetUsed)
        }
        if studioWorkspace.selectedModelId.isEmpty {
            studioWorkspace.selectedModelId = preferredDefaultModelId(for: request.task) ?? ""
        }
        syncWorkspaceDefaultsForTask()
    }

    public func resetStudioDraft(task: ProductTask? = nil) {
        let nextTask = task ?? studioWorkspace.task
        let workspaceId = studioWorkspace.workspaceId
        studioWorkspace = StudioWorkspaceDraft(workspaceId: workspaceId, task: nextTask)
        studioWorkspace.selectedModelId = preferredDefaultModelId(for: nextTask) ?? ""
        studioWorkspace.artifactFormat = nextTask.category == .image ? "png" : "mp4"
        syncWorkspaceDefaultsForTask()
    }

    public func syncWorkspaceDefaultsForTask() {
        let compatibleItems = catalog.items(for: studioWorkspace.task)
        let selectedItem = compatibleItems.first(where: { $0.modelId == studioWorkspace.selectedModelId })
        if
            studioWorkspace.selectedModelId.isEmpty
            || selectedItem == nil
            || (selectedItem?.installed == false && compatibleItems.contains(where: \.installed))
        {
            studioWorkspace.selectedModelId = preferredDefaultModelId(for: studioWorkspace.task) ?? ""
        }
        let expectedArtifactFormat = studioWorkspace.task.category == .image ? "png" : "mp4"
        if studioWorkspace.artifactFormat.isEmpty || studioWorkspace.artifactFormat != expectedArtifactFormat {
            studioWorkspace.artifactFormat = expectedArtifactFormat
        }
        if !studioWorkspace.useCustomSettings {
            let recommended = recommendedSettings(
                for: studioWorkspace.task,
                qualityPreset: studioWorkspace.qualityPreset,
                aspectPreset: studioWorkspace.aspectPreset,
                durationPreset: studioWorkspace.durationPreset,
                selectedModel: selectedItem
            )
            studioWorkspace.manualWidth = Double(recommended.width)
            studioWorkspace.manualHeight = Double(recommended.height)
            studioWorkspace.manualFrames = Double(recommended.numFrames)
            studioWorkspace.manualFps = Double(recommended.fps)
            studioWorkspace.manualSteps = Double(recommended.numInferenceSteps)
            studioWorkspace.manualGuidance = recommended.guidanceScale
        }
        normalizeWorkspaceReferences()
        applyFixedModelConstraints(for: selectedItem)
        if
            let selectedPackId = studioWorkspace.selectedPackId,
            !packCatalog.contains(where: {
                $0.id == selectedPackId
                    && $0.family == selectedItem?.family
                    && $0.supports(task: studioWorkspace.task)
            })
        {
            studioWorkspace.selectedPackId = nil
        }
        if studioWorkspace.selectedPackId == nil {
            studioWorkspace.selectedPackId = defaultPackId(for: studioWorkspace.selectedModelId, task: studioWorkspace.task)
        }
    }

    public func defaultPackId(for modelId: String, task: ProductTask) -> String? {
        guard let family = selectedModelFamily(for: modelId) else { return nil }
        return packCatalog.first(where: { $0.family == family && $0.supports(task: task) })?.id
    }

    public func preferredDefaultModelId(for task: ProductTask) -> String? {
        catalog.defaultModel(for: task)?.modelId
    }

    public func selectedModelFamily(for modelId: String) -> String? {
        catalog.items.first(where: { $0.modelId == modelId })?.family
    }

    public func selectedPack(for draft: StudioWorkspaceDraft? = nil) -> PackRecord? {
        let current = draft ?? studioWorkspace
        guard let selectedPackId = current.selectedPackId else { return nil }
        return packCatalog.first(where: { $0.id == selectedPackId })
    }

    public func prepareRunContext(
        task: ProductTask,
        title: String,
        sourceAssetIds: [String]
    ) -> WorkflowContextMetadata {
        WorkflowContextMetadata(
            workspaceId: studioWorkspace.workspaceId,
            collectionId: nil,
            runGroupId: UUID().uuidString,
            sourceAssetIds: sourceAssetIds,
            intentLabel: title.isEmpty ? task.title : title,
            presetId: studioWorkspace.aspectPreset.rawValue.lowercased()
        )
    }

    public func studioPlanningIntent(for draft: StudioWorkspaceDraft? = nil) -> WorkflowIntent? {
        let current = draft ?? studioWorkspace
        guard let selectedModel = selectedModel(for: current), selectedModel.installed else {
            return nil
        }
        let settings = resolvedSettings(for: current)
        let references = planningReferences(for: current, settings: settings)
        let familyExtensions = studioFamilyExtensions(for: current)
        var params: JSONMap = [
            "width": .integer(settings.width),
            "height": .integer(settings.height),
            "num_inference_steps": .integer(settings.numInferenceSteps),
            "guidance_scale": .number(settings.guidanceScale),
        ]
        if current.task.category == .video {
            params["num_frames"] = .integer(settings.numFrames)
            params["fps"] = .integer(settings.fps)
            params["regenerate_video"] = .bool(true)
            params["regenerate_audio"] = .bool(true)
        }
        if let seedValue = Int(current.seed) {
            params["seed"] = .integer(seedValue)
        }
        return WorkflowIntent(
            modelId: selectedModel.modelId,
            prompt: current.prompt,
            task: current.task.rawValue,
            negativePrompt: current.negativePrompt.isEmpty ? nil : current.negativePrompt,
            references: references,
            params: params,
            output: JobOutputPolicy(artifactFormat: current.artifactFormat),
            preferences: WorkflowPreferences(quality: current.qualityPreset.quality),
            extensions: namespacedExtensions(
                for: selectedModel.modelId,
                familyExtensions: familyExtensions
            )
        )
    }

    public func selectedModel(for draft: StudioWorkspaceDraft? = nil) -> ModelCatalogItem? {
        let current = draft ?? studioWorkspace
        let compatibleItems = catalog.items(for: current.task)
        return compatibleItems.first(where: { $0.modelId == current.selectedModelId })
            ?? compatibleItems.first
    }

    public var currentStudioReferenceRequirements: [WorkflowReferenceRequirement] {
        referenceRequirements(for: studioWorkspace)
    }

    public func resolvedSettings(for draft: StudioWorkspaceDraft? = nil) -> StudioResolvedSettings {
        let current = draft ?? studioWorkspace
        let selectedItem = selectedModel(for: current)
        let baseSettings: StudioResolvedSettings
        if current.useCustomSettings {
            baseSettings = StudioResolvedSettings(
                width: Int(current.manualWidth),
                height: Int(current.manualHeight),
                numFrames: Int(current.manualFrames),
                fps: Int(current.manualFps),
                numInferenceSteps: Int(current.manualSteps),
                guidanceScale: current.manualGuidance
            )
        } else {
            baseSettings = recommendedSettings(
                for: current.task,
                qualityPreset: current.qualityPreset,
                aspectPreset: current.aspectPreset,
                durationPreset: current.durationPreset,
                selectedModel: selectedItem
            )
        }

        guard let constraints = selectedItem?.capability?.constraints else {
            return baseSettings
        }
        return StudioResolvedSettings(
            width: baseSettings.width,
            height: baseSettings.height,
            numFrames: baseSettings.numFrames,
            fps: baseSettings.fps,
            numInferenceSteps: constraints.object("num_inference_steps")?.integer("fixed")
                ?? baseSettings.numInferenceSteps,
            guidanceScale: constraints.object("guidance_scale")?.double("fixed")
                ?? baseSettings.guidanceScale
        )
    }

    public func referenceKind(for asset: LibraryAsset, draft: StudioWorkspaceDraft? = nil) -> WorkflowReferenceKind? {
        let current = draft ?? studioWorkspace
        let requirements = referenceRequirements(for: current)
        guard !requirements.isEmpty else { return nil }
        let kind: WorkflowReferenceKind
        switch asset.kind {
        case .image:
            kind = .image
        case .video:
            kind = .video
        case .audio:
            kind = .audio
        case .other:
            return nil
        }
        return requirements.contains(where: { $0.kind == kind }) ? kind : nil
    }

    public func recommendedSettings(
        for task: ProductTask,
        qualityPreset: QualityPreset,
        aspectPreset: AspectPreset,
        durationPreset: DurationPreset,
        selectedModel: ModelCatalogItem?
    ) -> StudioResolvedSettings {
        let dimensions = aspectPreset.dimensions(for: task.category)
        let imageStepsByFamily: [String: (Int, Int, Int)] = [
            "flux2": (4, 8, 12),
            "qwen_image": (12, 24, 36),
            "z_image": (6, 12, 20),
        ]
        let defaultImageSteps = imageStepsByFamily[selectedModel?.family ?? ""] ?? (8, 16, 24)
        let imageSteps: Int = switch qualityPreset {
        case .draft: defaultImageSteps.0
        case .standard: defaultImageSteps.1
        case .cinematic: defaultImageSteps.2
        }
        let videoSteps: Int = switch qualityPreset {
        case .draft: 18
        case .standard: 28
        case .cinematic: 40
        }
        let guidance = selectedModel?.capability?.constraints.object("guidance_scale")?.double("fixed")
            ?? (task.category == .image ? 4.0 : 3.0)

        return StudioResolvedSettings(
            width: dimensions.width,
            height: dimensions.height,
            numFrames: task.category == .video ? durationPreset.numFrames : 1,
            fps: task.category == .video ? 24 : 1,
            numInferenceSteps: task.category == .image ? imageSteps : videoSteps,
            guidanceScale: guidance
        )
    }

    public func hasRecommendedReset(for draft: StudioWorkspaceDraft? = nil) -> Bool {
        let current = draft ?? studioWorkspace
        guard current.useCustomSettings else { return false }
        let recommended = recommendedSettings(
            for: current.task,
            qualityPreset: current.qualityPreset,
            aspectPreset: current.aspectPreset,
            durationPreset: current.durationPreset,
            selectedModel: selectedModel(for: current)
        )
        let currentSettings = StudioResolvedSettings(
            width: Int(current.manualWidth),
            height: Int(current.manualHeight),
            numFrames: Int(current.manualFrames),
            fps: Int(current.manualFps),
            numInferenceSteps: Int(current.manualSteps),
            guidanceScale: current.manualGuidance
        )
        return currentSettings != recommended
    }

    public func resetWorkspaceToRecommended() {
        studioWorkspace.useCustomSettings = false
        syncWorkspaceDefaultsForTask()
    }

    public func toggleFavorite(assetId: String) {
        var record = assetRecords[assetId] ?? AssetRecord(id: assetId)
        record.isFavorite.toggle()
        assetRecords[assetId] = record
    }

    public func markAssetUsed(_ assetId: String) {
        var record = assetRecords[assetId] ?? AssetRecord(id: assetId)
        record.lastUsedAt = .now
        assetRecords[assetId] = record
    }

    public func createCollection(named title: String) {
        let normalized = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        collections.insert(CollectionRecord(title: normalized), at: 0)
    }

    @discardableResult
    public func createWorkspace(named title: String? = nil) -> WorkspaceRecord {
        let trimmed = title?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedTitle = (trimmed?.isEmpty == false ? trimmed : nil) ?? "New Project"
        let workspace = WorkspaceRecord(title: resolvedTitle)
        workspaces.insert(workspace, at: 0)
        selectWorkspace(workspace.id)
        return workspace
    }

    public func selectWorkspace(_ workspaceId: String) {
        activeWorkspaceId = workspaceId
        selectedLibraryWorkspaceId = workspaceId
        ensureWorkspaceExists()
    }

    public func showLibraryBrowser() {
        selectedLibraryWorkspaceId = nil
    }

    public func toggleAsset(_ assetId: String, in collectionId: String) {
        var record = assetRecords[assetId] ?? AssetRecord(id: assetId)
        if record.collectionIds.contains(collectionId) {
            record.collectionIds.removeAll { $0 == collectionId }
        } else {
            record.collectionIds.append(collectionId)
            record.collectionIds.sort()
        }
        assetRecords[assetId] = record
    }

    public func removeImportedAssetFromLibrary(_ assetId: String) {
        assetRecords.removeValue(forKey: assetId)
    }

    func recordAcceptedRunGroup(
        runGroupId: String,
        task: ProductTask,
        title: String,
        context: WorkflowContextMetadata?,
        jobId: String,
        variationCount: Int
    ) {
        ensureWorkspaceExists()
        if let index = runGroups.firstIndex(where: { $0.id == runGroupId }) {
            if !runGroups[index].jobIds.contains(jobId) {
                runGroups[index].jobIds.append(jobId)
            }
            runGroups[index].state = .running
            runGroups[index].updatedAt = .now
            runGroups[index].variationCount = max(runGroups[index].variationCount, variationCount)
            studioWorkspace.lastActiveRunGroupId = runGroupId
            lastSubmittedJobId = jobId
            renameWorkspaceIfPlaceholder(
                workspaceId: runGroups[index].workspaceId,
                using: context?.intentLabel ?? title
            )
            return
        }

        let group = RunGroupRecord(
            id: runGroupId,
            workspaceId: context?.workspaceId ?? activeWorkspaceId,
            collectionId: context?.collectionId,
            task: task,
            title: context?.intentLabel ?? title,
            sourceAssetIds: context?.sourceAssetIds ?? [],
            jobIds: [jobId],
            assetIds: [],
            variationCount: max(variationCount, 1),
            createdAt: .now,
            updatedAt: .now,
            state: .running
        )
        runGroups.insert(group, at: 0)
        studioWorkspace.lastActiveRunGroupId = runGroupId
        lastSubmittedJobId = jobId
        renameWorkspaceIfPlaceholder(
            workspaceId: group.workspaceId,
            using: context?.intentLabel ?? title
        )
    }

    func syncRunGroupsFromJobs() {
        let assets = libraryAssets
        let jobsByRunGroup = Dictionary(grouping: jobs) { $0.request.context?.runGroupId }
            .compactMapValues { groupJobs in
                groupJobs.filter { $0.request.context?.runGroupId != nil }
            }

        var knownGroupIds = Set(runGroups.map(\.id))
        let reconstructedGroups = jobsByRunGroup.compactMap { runGroupId, groupJobs -> RunGroupRecord? in
            guard
                let runGroupId,
                !knownGroupIds.contains(runGroupId),
                let seedJob = groupJobs.sorted(by: { $0.createdAt < $1.createdAt }).first,
                let task = ProductTask.from(rawTask: seedJob.request.task)
            else {
                return nil
            }

            knownGroupIds.insert(runGroupId)
            let context = seedJob.request.context
            let relatedAssets = assets.filter { $0.runGroupId == runGroupId }
            return RunGroupRecord(
                id: runGroupId,
                workspaceId: context?.workspaceId ?? activeWorkspaceId,
                collectionId: context?.collectionId,
                task: task,
                title: context?.intentLabel ?? task.title,
                sourceAssetIds: context?.sourceAssetIds ?? [],
                jobIds: groupJobs.map(\.jobId),
                assetIds: relatedAssets.map(\.id),
                variationCount: max(max(groupJobs.count, relatedAssets.count), 1),
                createdAt: groupJobs.map(\.createdAt).min() ?? .now,
                updatedAt: groupJobs.map(\.updatedAt).max() ?? .now,
                state: .queued
            )
        }
        if !reconstructedGroups.isEmpty {
            runGroups = (reconstructedGroups + runGroups).sorted { $0.updatedAt > $1.updatedAt }
        }

        for index in runGroups.indices {
            let groupId = runGroups[index].id
            let relatedJobs = jobs.filter { $0.request.context?.runGroupId == groupId }
            runGroups[index].jobIds = relatedJobs.map(\.jobId)
            runGroups[index].assetIds = assets.filter { $0.runGroupId == groupId }.map(\.id)
            runGroups[index].variationCount = max(max(runGroups[index].variationCount, relatedJobs.count), max(runGroups[index].assetIds.count, 1))
            if relatedJobs.isEmpty {
                runGroups[index].state = assets.contains(where: { $0.runGroupId == groupId }) ? .completed : .queued
            } else {
                runGroups[index].state = state(for: relatedJobs, assetCount: runGroups[index].assetIds.count)
            }
            runGroups[index].updatedAt = relatedJobs.map(\.updatedAt).max() ?? runGroups[index].updatedAt
        }
        runGroups.sort { $0.updatedAt > $1.updatedAt }
    }

    private func shouldIncludeInActivity(
        _ group: RunGroupRecord,
        retentionCutoff: Date
    ) -> Bool {
        switch group.state {
        case .queued, .running:
            return true
        case .failed:
            return !dismissedActivityRunGroupIds.contains(group.id)
        case .completed, .cancelled:
            return
                group.updatedAt >= retentionCutoff
                && !dismissedActivityRunGroupIds.contains(group.id)
        }
    }

    private func legacyActivityTitle(for job: JobRecord, task: ProductTask) -> String {
        if let prompt = job.request.inputs.string("prompt") {
            let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                if trimmed.count <= 72 {
                    return trimmed
                }
                let truncated = String(trimmed.prefix(69)).trimmingCharacters(in: .whitespacesAndNewlines)
                return "\(truncated)…"
            }
        }
        return task.title
    }

    private func normalizeWorkspaceReferences() {
        let requirements = referenceRequirements(for: studioWorkspace)
        let allowedKinds = Set(requirements.map(\.kind))
        let anyReferenceAssetIds = Set(
            libraryAssets
                .filter { $0.referenceKind != nil }
                .map(\.id)
        )
        let compatibleAssetIds = Set(
            libraryAssets
                .filter { asset in
                    guard let kind = referenceKind(for: asset) else { return false }
                    return allowedKinds.contains(kind)
                }
                .map(\.id)
        )
        var referenceIds = studioWorkspace.referenceAssetIds.filter { compatibleAssetIds.contains($0) }

        if allowedKinds.isEmpty {
            let fallbackIds = studioWorkspace.referenceAssetIds.filter { anyReferenceAssetIds.contains($0) }
            studioWorkspace.referenceAssetIds = Array(NSOrderedSet(array: fallbackIds)) as? [String] ?? fallbackIds
            return
        }

        if referenceIds.isEmpty,
           let selectedAssetId = studioWorkspace.selectedAssetId,
           compatibleAssetIds.contains(selectedAssetId)
        {
            referenceIds = [selectedAssetId]
        }

        referenceIds = trimmedReferenceIds(referenceIds, requirements: requirements)
        studioWorkspace.referenceAssetIds = Array(NSOrderedSet(array: referenceIds)) as? [String] ?? referenceIds
    }

    private func applyFixedModelConstraints(for selectedItem: ModelCatalogItem?) {
        guard let constraints = selectedItem?.capability?.constraints else {
            return
        }
        if let fixedSteps = constraints.object("num_inference_steps")?.integer("fixed") {
            studioWorkspace.manualSteps = Double(fixedSteps)
        }
        if let fixedGuidance = constraints.object("guidance_scale")?.double("fixed") {
            studioWorkspace.manualGuidance = fixedGuidance
        }
    }

    private func trimmedReferenceIds(
        _ referenceIds: [String],
        requirements: [WorkflowReferenceRequirement]
    ) -> [String] {
        guard !requirements.isEmpty else { return [] }
        var remainingByKind = Dictionary(
            uniqueKeysWithValues: requirements.compactMap { requirement in
                requirement.maximumCount.map { (requirement.kind, $0) }
            }
        )
        var trimmed: [String] = []
        for referenceId in referenceIds {
            guard let asset = libraryAssets.first(where: { $0.id == referenceId }),
                  let kind = referenceKind(for: asset) else {
                continue
            }
            if let remaining = remainingByKind[kind] {
                guard remaining > 0 else { continue }
                remainingByKind[kind] = remaining - 1
            }
            trimmed.append(referenceId)
        }
        return trimmed
    }

    private func referenceRequirements(for draft: StudioWorkspaceDraft) -> [WorkflowReferenceRequirement] {
        if let studioPlanResult, studioPlanMatchesDraft(studioPlanResult, draft: draft) {
            return studioPlanResult.readiness.referenceRequirements
        }
        return fallbackReferenceRequirements(for: draft.task)
    }

    private func studioPlanMatchesDraft(
        _ result: WorkflowPlanResult,
        draft: StudioWorkspaceDraft
    ) -> Bool {
        result.plan.selectedTask == draft.task.rawValue
            || result.presentation.selectedTask == draft.task.rawValue
    }

    private func fallbackReferenceRequirements(for task: ProductTask) -> [WorkflowReferenceRequirement] {
        switch task {
        case .imageEdit:
            return [
                WorkflowReferenceRequirement(
                    kind: .image,
                    minimumCount: 1,
                    maximumCount: 1,
                    acceptedRoles: ["reference"],
                    description: "Choose at least one image to edit."
                )
            ]
        case .videoConditionImage, .videoInterpolate:
            return [
                WorkflowReferenceRequirement(
                    kind: .image,
                    minimumCount: task == .videoInterpolate ? 2 : 1,
                    maximumCount: task == .videoInterpolate ? 2 : 1,
                    acceptedRoles: ["reference"],
                    description: task == .videoInterpolate
                        ? "Choose a start and end frame to blend."
                        : "Choose an image to animate."
                )
            ]
        case .videoConditionAudio:
            return [
                WorkflowReferenceRequirement(
                    kind: .audio,
                    minimumCount: 1,
                    maximumCount: 1,
                    acceptedRoles: ["reference"],
                    description: "Choose an audio guide for the clip."
                )
            ]
        case .videoConditionVideo, .videoRetake:
            return [
                WorkflowReferenceRequirement(
                    kind: .video,
                    minimumCount: 1,
                    maximumCount: 1,
                    acceptedRoles: ["reference"],
                    description: task == .videoRetake
                        ? "Choose the source video to retake."
                        : "Choose a guide video."
                )
            ]
        case .imageGenerate, .videoGenerate:
            return []
        }
    }

    private func studioFamilyExtensions(for draft: StudioWorkspaceDraft) -> JSONMap {
        var familyExtensions = selectedPack(for: draft)?.familyExtensions ?? [:]
        if let workflowVariant = selectedPack(for: draft)?.workflowVariant {
            familyExtensions["workflow_variant"] = .string(workflowVariant)
        }
        if let controlVariant = selectedPack(for: draft)?.controlVariant {
            familyExtensions["control_variant"] = .string(controlVariant)
        }
        if let conditioningAttentionStrength = selectedPack(for: draft)?.conditioningAttentionStrength {
            familyExtensions["conditioning_attention_strength"] = .number(conditioningAttentionStrength)
        }
        return familyExtensions
    }

    private func planningReferences(
        for draft: StudioWorkspaceDraft,
        settings: StudioResolvedSettings
    ) -> [WorkflowReference] {
        let referenceAssets = libraryAssets.filter { draft.referenceAssetIds.contains($0.id) }
        let imageFrameIndices = imageFrameIndicesForPlanning(
            count: referenceAssets.filter(\.isImage).count,
            numFrames: settings.numFrames,
            task: draft.task
        )
        var imageIndex = 0
        var references: [WorkflowReference] = []
        for asset in referenceAssets {
            guard let kind = referenceKind(for: asset, draft: draft) else { continue }
            var metadata: JSONMap = [:]
            switch kind {
            case .image:
                metadata["frame_index"] = .integer(imageFrameIndices[imageIndex])
                metadata["strength"] = .number(1.0)
                imageIndex += 1
            case .video, .lora:
                metadata["strength"] = .number(1.0)
            case .audio:
                metadata["start_time_seconds"] = .number(0.0)
            }
            references.append(
                WorkflowReference(
                    inputHandle: nil,
                    kind: kind,
                    role: "reference",
                    metadata: metadata
                )
            )
        }
        return references
    }

    private func imageFrameIndicesForPlanning(
        count: Int,
        numFrames: Int,
        task: ProductTask
    ) -> [Int] {
        guard count > 0 else { return [] }
        guard task == .videoInterpolate, count > 1 else {
            return Array(repeating: 0, count: count)
        }
        let maxFrame = max(numFrames - 1, 0)
        if count == 2 {
            return [0, maxFrame]
        }
        let step = maxFrame / max(count - 1, 1)
        return (0..<count).map { min($0 * step, maxFrame) }
    }

    private func state(for jobs: [JobRecord], assetCount: Int) -> RunGroupState {
        if jobs.contains(where: { $0.state == .failed }) {
            return .failed
        }
        if jobs.contains(where: { $0.state == .cancelled }) && jobs.allSatisfy(\.state.isTerminal) {
            return .cancelled
        }
        if jobs.allSatisfy(\.state.isTerminal), !jobs.isEmpty {
            return .completed
        }
        if jobs.contains(where: { !$0.state.isTerminal }) {
            return .running
        }
        if assetCount > 0 {
            return .completed
        }
        return .queued
    }

    private func ensureWorkspaceExists() {
        if workspaces.contains(where: { $0.id == activeWorkspaceId }) == false {
            workspaces.append(WorkspaceRecord(id: activeWorkspaceId, title: "New Project"))
        }
        if studioWorkspace.workspaceId != activeWorkspaceId {
            studioWorkspace.workspaceId = activeWorkspaceId
        }
        if let index = workspaces.firstIndex(where: { $0.id == activeWorkspaceId }) {
            workspaces[index].lastOpenedAt = .now
        }
    }

    private func renameWorkspaceIfPlaceholder(workspaceId: String, using title: String) {
        guard let index = workspaces.firstIndex(where: { $0.id == workspaceId }) else {
            return
        }
        let currentTitle = workspaces[index].title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard currentTitle == "Current Workspace" || currentTitle == "New Project" else {
            return
        }
        let normalized = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            return
        }
        let headline: String
        if normalized.count <= 48 {
            headline = normalized
        } else {
            let cutoff = normalized.index(normalized.startIndex, offsetBy: 45)
            let truncated = String(normalized[..<cutoff]).trimmingCharacters(in: .whitespacesAndNewlines)
            headline = "\(truncated)…"
        }
        workspaces[index].title = headline
    }
}
