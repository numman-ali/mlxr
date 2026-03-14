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

    public func persistPresentationState() {
        do {
            try workspaceStateStore.persist(
                AppPresentationState(
                    hasCompletedModelSetup: hasCompletedOnboarding,
                    activeWorkspaceId: activeWorkspaceId,
                    workspaces: workspaces,
                    collections: collections,
                    runGroups: runGroups,
                    assets: Array(assetRecords.values).sorted { $0.id < $1.id },
                    workspaceDraft: studioWorkspace
                )
            )
        } catch {
            settingsError = error.localizedDescription
        }
    }

    public func openStudio(
        task: ProductTask,
        focusedAssetId: String? = nil,
        referenceAssetIds: [String] = [],
        prompt: String? = nil
    ) {
        ensureWorkspaceExists()
        studioWorkspace.task = task
        if let resolvedPrompt = prompt ?? focusedAssetId.flatMap({ assetId in
            libraryAssets.first(where: { $0.id == assetId })?.prompt
        }), !resolvedPrompt.isEmpty {
            studioWorkspace.prompt = resolvedPrompt
        }
        if let focusedAssetId {
            studioWorkspace.selectedAssetId = focusedAssetId
            studioWorkspace.preferredDisplayedAssetId = focusedAssetId
            markAssetUsed(focusedAssetId)
        }
        if !referenceAssetIds.isEmpty {
            studioWorkspace.referenceAssetIds = referenceAssetIds
            referenceAssetIds.forEach(markAssetUsed)
        }
        if studioWorkspace.selectedModelId.isEmpty {
            studioWorkspace.selectedModelId = preferredDefaultModelId(for: task) ?? ""
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
                selectedFamily: selectedModelFamily(for: studioWorkspace.selectedModelId)
            )
            studioWorkspace.manualWidth = Double(recommended.width)
            studioWorkspace.manualHeight = Double(recommended.height)
            studioWorkspace.manualFrames = Double(recommended.numFrames)
            studioWorkspace.manualFps = Double(recommended.fps)
            studioWorkspace.manualSteps = Double(recommended.numInferenceSteps)
            studioWorkspace.manualGuidance = recommended.guidanceScale
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
        let preferredIds: [String]
        switch task {
        case .imageGenerate:
            preferredIds = ["z-image-turbo-local", "qwen-image-local", "flux2-klein-9b-local"]
        case .imageEdit:
            preferredIds = ["qwen-image-edit-local", "flux2-klein-9b-local", "qwen-image-local"]
        default:
            preferredIds = ["ltx-2.3-fast-local"]
        }

        let compatibleItems = catalog.items(for: task)
        for modelId in preferredIds {
            if compatibleItems.contains(where: { $0.modelId == modelId && $0.installed }) {
                return modelId
            }
        }
        for modelId in preferredIds {
            if compatibleItems.contains(where: { $0.modelId == modelId }) {
                return modelId
            }
        }
        return compatibleItems.first(where: \.installed)?.modelId ?? compatibleItems.first?.modelId
    }

    public func selectedModelFamily(for modelId: String) -> String? {
        catalog.items.first(where: { $0.modelId == modelId })?.family
    }

    public func selectedPack(for draft: StudioWorkspaceDraft? = nil) -> PackRecord? {
        let current = draft ?? studioWorkspace
        guard let selectedPackId = current.selectedPackId else { return nil }
        return packCatalog.first(where: { $0.id == selectedPackId })
    }

    public func recommendedSettings(
        for task: ProductTask,
        qualityPreset: QualityPreset,
        aspectPreset: AspectPreset,
        durationPreset: DurationPreset,
        selectedFamily: String?
    ) -> StudioResolvedSettings {
        let dimensions = aspectPreset.dimensions(for: task.category)
        let imageStepsByFamily: [String: (Int, Int, Int)] = [
            "flux2": (4, 8, 12),
            "qwen_image": (12, 24, 36),
            "z_image": (6, 12, 20),
        ]
        let defaultImageSteps = imageStepsByFamily[selectedFamily ?? ""] ?? (8, 16, 24)
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
        let guidance = task.category == .image ? 4.0 : 3.0

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
            selectedFamily: selectedModelFamily(for: current.selectedModelId)
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

    func createRunGroup(task: ProductTask, title: String, sourceAssetIds: [String], variationCount: Int) -> RunGroupRecord {
        let group = RunGroupRecord(
            workspaceId: activeWorkspaceId,
            collectionId: nil,
            task: task,
            title: title,
            sourceAssetIds: sourceAssetIds,
            variationCount: variationCount,
            state: .queued
        )
        runGroups.insert(group, at: 0)
        studioWorkspace.lastActiveRunGroupId = group.id
        return group
    }

    func recordSubmittedJob(_ jobId: String, in runGroupId: String) {
        guard let index = runGroups.firstIndex(where: { $0.id == runGroupId }) else { return }
        if !runGroups[index].jobIds.contains(jobId) {
            runGroups[index].jobIds.append(jobId)
        }
        runGroups[index].state = .running
        runGroups[index].updatedAt = .now
        lastSubmittedJobId = jobId
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
            if relatedJobs.contains(where: { $0.state == .failed }) {
                runGroups[index].state = .failed
            } else if relatedJobs.contains(where: { $0.state == .cancelled }) && relatedJobs.allSatisfy(\.state.isTerminal) {
                runGroups[index].state = .cancelled
            } else if relatedJobs.allSatisfy(\.state.isTerminal), !relatedJobs.isEmpty {
                runGroups[index].state = .completed
            } else if relatedJobs.contains(where: { !$0.state.isTerminal }) {
                runGroups[index].state = .running
            } else if relatedJobs.isEmpty {
                runGroups[index].state = assets.contains(where: { $0.runGroupId == groupId }) ? .completed : .queued
            } else {
                runGroups[index].state = .queued
            }
            runGroups[index].updatedAt = relatedJobs.map(\.updatedAt).max() ?? runGroups[index].updatedAt
        }
        runGroups.sort { $0.updatedAt > $1.updatedAt }
    }

    private func ensureWorkspaceExists() {
        if workspaces.contains(where: { $0.id == activeWorkspaceId }) == false {
            workspaces.append(WorkspaceRecord(id: activeWorkspaceId, title: "Current Workspace"))
        }
        if studioWorkspace.workspaceId != activeWorkspaceId {
            studioWorkspace.workspaceId = activeWorkspaceId
        }
        if let index = workspaces.firstIndex(where: { $0.id == activeWorkspaceId }) {
            workspaces[index].lastOpenedAt = .now
        }
    }
}
