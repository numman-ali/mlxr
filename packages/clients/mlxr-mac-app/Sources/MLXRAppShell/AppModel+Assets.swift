import Foundation
import MLXRAppDomain

@MainActor
extension MLXRAppModel {
    public var libraryAssets: [LibraryAsset] {
        libraryAssetsCache
    }

    public var recentLibraryAssets: [LibraryAsset] {
        recentLibraryAssetsCache
    }

    public var latestGeneratedAsset: LibraryAsset? {
        latestGeneratedAssetCache
    }

    public func importExternalAssets(from urls: [URL], workspaceId: String) async -> [ImportedAssetRecord] {
        guard !urls.isEmpty else { return [] }
        do {
            let updated = try importedAssetStore.importFiles(
                at: urls,
                existing: importedAssets,
                workspaceId: workspaceId
            )
            let existingIds = Set(importedAssets.map { $0.id })
            let createdIds = Set(updated.map { $0.id }).subtracting(existingIds)
            importedAssets = updated
            globalError = nil
            let createdAssets = importedAssets.filter { createdIds.contains($0.id) }
            guard !createdAssets.isEmpty else {
                return []
            }
            selectWorkspace(
                workspaceId,
                clearingDraftContext: creationDraft.workspaceId != workspaceId
            )
            return createdAssets
        } catch {
            globalError = error.localizedDescription
            return []
        }
    }

    public func removeImportedAsset(assetId: String) async {
        do {
            importedAssets = try importedAssetStore.remove(recordId: assetId, existing: importedAssets)
            removeImportedAssetFromLibrary(assetId)
            globalError = nil
        } catch {
            globalError = error.localizedDescription
        }
    }

    public func resolvedURL(for asset: LibraryAsset) async -> URL? {
        if let importedAsset = asset.importedAsset {
            return importedAssetStore.fileURL(for: importedAsset)
        }
        guard let artifact = asset.artifact else {
            return nil
        }
        return await cachedOutputURL(for: artifact)
    }

    func rebuildDerivedLibraryState() {
        let entries = LibraryEntry.flatten(jobs: jobs)
        let generatedAssets = LibraryAsset.fromGeneratedEntries(entries, metadataById: assetRecords)
        let importedLibraryAssets = LibraryAsset.fromImported(importedAssets, metadataById: assetRecords)
        let libraryAssets = (generatedAssets + importedLibraryAssets).sorted {
            let lhsSort = $0.lastUsedAt ?? $0.createdAt
            let rhsSort = $1.lastUsedAt ?? $1.createdAt
            if lhsSort != rhsSort {
                return lhsSort > rhsSort
            }
            return $0.createdAt > $1.createdAt
        }
        libraryEntriesCache = entries
        libraryAssetsCache = libraryAssets
        recentLibraryAssetsCache = Array(libraryAssets.prefix(12))
        latestGeneratedAssetCache = libraryAssets.first(where: \.isGenerated)
        hasContentCache = !libraryAssets.isEmpty
        recentCompletedAssetsCache = libraryAssets.filter(\.isGenerated).sorted { $0.createdAt > $1.createdAt }
        latestSubmittedAssetCache =
            if let lastSubmittedJobId {
                libraryAssets.first { $0.jobId == lastSubmittedJobId }
            } else {
                nil
            }
        activeJobCountCache = jobs.reduce(into: 0) { count, job in
            if !job.state.isTerminal {
                count += 1
            }
        }
        totalCreationCountCache = jobs.reduce(into: 0) { count, job in
            if job.state == .completed {
                count += 1
            }
        }
        rebuildPreferredWorkspaceCache()
        rebuildActivityRunGroupsCache()
        bumpLibraryPresentationRevision()
    }

    func rebuildPreferredWorkspaceCache() {
        preferredLibraryWorkspaceIdCache =
            if let selectedLibraryWorkspaceId,
               libraryAssetsCache.contains(where: { cachedResolvedWorkspaceId(for: $0) == selectedLibraryWorkspaceId }) {
                selectedLibraryWorkspaceId
            } else if libraryAssetsCache.contains(where: { cachedResolvedWorkspaceId(for: $0) == activeWorkspaceId }) {
                activeWorkspaceId
            } else if libraryAssetsCache.contains(where: { cachedResolvedWorkspaceId(for: $0) == defaultWorkspaceId }) {
                defaultWorkspaceId
            } else {
                libraryAssetsCache.first.map(cachedResolvedWorkspaceId(for:))
            }
    }

    func rebuildActivityRunGroupsCache() {
        let assetsByJobId = Dictionary(
            grouping: libraryAssetsCache.compactMap { asset -> LibraryAsset? in
                guard asset.jobId != nil else { return nil }
                return asset
            },
            by: { $0.jobId ?? "" }
        )
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
            let relatedAssets = assetsByJobId[job.jobId] ?? []
            let group = RunGroupRecord(
                id: syntheticId,
                workspaceId: relatedAssets.first.map(cachedResolvedWorkspaceId(for:)) ?? defaultWorkspaceId,
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
            return shouldIncludeInActivity(group) ? group : nil
        }
        let persistedGroups = runGroups.filter { shouldIncludeInActivity($0) }
        activityRunGroupsCache = (persistedGroups + legacyGroups).sorted { $0.updatedAt > $1.updatedAt }
    }

    private func cachedResolvedWorkspaceId(for asset: LibraryAsset) -> String {
        if let workspaceId = asset.workspaceId {
            return workspaceId
        }
        if let runGroupId = asset.runGroupId,
           let runGroup = runGroupsByIdCache[runGroupId]
        {
            return runGroup.workspaceId
        }
        if let importedWorkspaceId = asset.importedAsset?.workspaceId {
            return importedWorkspaceId
        }
        return defaultWorkspaceId
    }
}
