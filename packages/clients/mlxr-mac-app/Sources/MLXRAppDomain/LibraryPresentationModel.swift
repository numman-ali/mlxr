import Foundation

public struct LibraryFilterState: Hashable, Sendable {
    public var selectedFilter: LibraryAssetFilter
    public var selectedModelId: String
    public var selectedTaskRaw: String
    public var selectedSort: AssetSortMode
    public var favoritesOnly: Bool
    public var selectedCollectionId: String?
    public var query: String

    public init(
        selectedFilter: LibraryAssetFilter = .all,
        selectedModelId: String = "all",
        selectedTaskRaw: String = "all",
        selectedSort: AssetSortMode = .newest,
        favoritesOnly: Bool = false,
        selectedCollectionId: String? = nil,
        query: String = ""
    ) {
        self.selectedFilter = selectedFilter
        self.selectedModelId = selectedModelId
        self.selectedTaskRaw = selectedTaskRaw
        self.selectedSort = selectedSort
        self.favoritesOnly = favoritesOnly
        self.selectedCollectionId = selectedCollectionId
        self.query = query
    }
}

public struct ProjectSummaryPresentation: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let subtitle: String
    public let heroAsset: LibraryAsset?
    public let assetCount: Int
    public let imageCount: Int
    public let videoCount: Int
    public let audioCount: Int
    public let runGroupCount: Int
    public let updatedAt: Date
    public let isActive: Bool

    public var hasContent: Bool {
        assetCount > 0
    }
}

public struct LibraryAssetGroupPresentation: Identifiable, Hashable, Sendable {
    public let id: String
    public let runGroupId: String?
    public let title: String
    public let summary: String
    public let assets: [LibraryAsset]
    public let primaryAsset: LibraryAsset
    public let runState: RunGroupState?
    public let createdAt: Date
    public let updatedAt: Date
    public let variationCount: Int

    public var assetCount: Int {
        assets.count
    }

    public var isImportedOnly: Bool {
        assets.allSatisfy(\.isImported)
    }
}

public struct LibraryPresentationModel: Sendable {
    private let assets: [LibraryAsset]
    private let workspacesById: [String: WorkspaceRecord]
    private let runGroupsById: [String: RunGroupRecord]
    private let defaultWorkspaceId: String

    public let collections: [CollectionRecord]
    public let filters: LibraryFilterState
    public let selectedWorkspaceId: String?

    public init(
        assets: [LibraryAsset],
        workspaces: [WorkspaceRecord],
        runGroups: [RunGroupRecord],
        collections: [CollectionRecord],
        filters: LibraryFilterState,
        selectedWorkspaceId: String?,
        defaultWorkspaceId: String
    ) {
        self.assets = assets
        self.workspacesById = Dictionary(uniqueKeysWithValues: workspaces.map { ($0.id, $0) })
        self.runGroupsById = Dictionary(uniqueKeysWithValues: runGroups.map { ($0.id, $0) })
        self.collections = collections
        self.filters = filters
        self.selectedWorkspaceId = selectedWorkspaceId
        self.defaultWorkspaceId = defaultWorkspaceId
    }

    public var selectedWorkspace: WorkspaceRecord? {
        guard let selectedWorkspaceId else { return nil }
        return workspacesById[selectedWorkspaceId]
    }

    public var modelOptions: [String] {
        Array(Set(filteredAssets.compactMap(\.modelId))).sorted()
    }

    public var taskOptions: [ProductTask] {
        Array(Set(filteredAssets.compactMap(\.task))).sorted { $0.title < $1.title }
    }

    public func count(for filter: LibraryAssetFilter) -> Int {
        assets.filter { filter.includes($0) }.count
    }

    public func count(for collectionId: String?) -> Int {
        guard let collectionId else { return assets.count }
        return assets.filter { $0.collectionIds.contains(collectionId) }.count
    }

    public var projectSummaries: [ProjectSummaryPresentation] {
        workspacesById.values.compactMap(projectSummary(for:))
            .filter(projectMatchesFilters(_:))
            .sorted { lhs, rhs in
                if lhs.updatedAt != rhs.updatedAt {
                    return lhs.updatedAt > rhs.updatedAt
                }
                return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
            }
    }

    public var filteredAssets: [LibraryAsset] {
        workspaceScopedAssets(selectedWorkspaceId)
            .filter(matchesFilters(_:))
            .sorted(by: sortComparator)
    }

    public var groups: [LibraryAssetGroupPresentation] {
        let groupedAssets = Dictionary(grouping: filteredAssets) { asset in
            asset.runGroupId ?? asset.id
        }
        return groupedAssets.compactMap { groupId, grouped in
            let sortedAssets = grouped.sorted(by: sortComparator)
            guard let primaryAsset = sortedAssets.first else {
                return nil
            }
            let runGroup = primaryAsset.runGroupId.flatMap { runGroupsById[$0] }
            return LibraryAssetGroupPresentation(
                id: groupId,
                runGroupId: runGroup?.id,
                title: runGroup?.title ?? primaryAsset.displayTitle,
                summary: runGroup?.task.title ?? primaryAsset.sourceSummary,
                assets: sortedAssets,
                primaryAsset: primaryAsset,
                runState: runGroup?.state,
                createdAt: sortedAssets.map(\.createdAt).min() ?? primaryAsset.createdAt,
                updatedAt: runGroup?.updatedAt ?? sortedAssets.map(\.createdAt).max() ?? primaryAsset.createdAt,
                variationCount: max(runGroup?.variationCount ?? 1, sortedAssets.count)
            )
        }
        .sorted { lhs, rhs in
            sortTimestamp(for: lhs) > sortTimestamp(for: rhs)
        }
    }

    public var orderedPrimaryAssetIds: [String] {
        groups.map { $0.primaryAsset.id }
    }

    public func group(containing assetId: String) -> LibraryAssetGroupPresentation? {
        groups.first { group in
            group.assets.contains(where: { $0.id == assetId })
        }
    }

    public func viewerAsset(for assetId: String?) -> LibraryAsset? {
        guard let assetId else { return nil }
        return filteredAssets.first(where: { $0.id == assetId })
            ?? assets.first(where: { $0.id == assetId })
    }

    public func projectSummary(id: String?) -> ProjectSummaryPresentation? {
        guard let id, let workspace = workspacesById[id] else { return nil }
        return projectSummary(for: workspace)
    }

    private func projectSummary(for workspace: WorkspaceRecord) -> ProjectSummaryPresentation? {
        let workspaceAssets = workspaceScopedAssets(workspace.id)
        guard !workspaceAssets.isEmpty || workspace.id == selectedWorkspaceId else {
            return nil
        }
        let heroAsset = workspaceAssets.sorted(by: sortComparator).first
        let latestPrompt = heroAsset?.promptHeadline ?? heroAsset?.displayTitle ?? "No results yet"
        let updatedAt = workspaceAssets
            .map { $0.lastUsedAt ?? $0.createdAt }
            .max()
            ?? workspace.lastOpenedAt
        let runGroupCount = Set(
            workspaceAssets.compactMap { asset in
                asset.runGroupId ?? asset.jobId
            }
        ).count
        return ProjectSummaryPresentation(
            id: workspace.id,
            title: workspace.title,
            subtitle: latestPrompt,
            heroAsset: heroAsset,
            assetCount: workspaceAssets.count,
            imageCount: workspaceAssets.filter(\.isImage).count,
            videoCount: workspaceAssets.filter(\.isVideo).count,
            audioCount: workspaceAssets.filter(\.isAudio).count,
            runGroupCount: runGroupCount,
            updatedAt: updatedAt,
            isActive: workspace.id == selectedWorkspaceId
        )
    }

    private func projectMatchesFilters(_ summary: ProjectSummaryPresentation) -> Bool {
        let trimmedQuery = normalizedQuery
        if trimmedQuery.isEmpty {
            return true
        }
        if summary.title.lowercased().contains(trimmedQuery) {
            return true
        }
        let workspaceAssets = workspaceScopedAssets(summary.id)
        return workspaceAssets.contains { $0.searchableText.contains(trimmedQuery) }
    }

    private func workspaceScopedAssets(_ workspaceId: String?) -> [LibraryAsset] {
        guard let workspaceId else {
            return assets
        }
        return assets.filter { effectiveWorkspaceId(for: $0) == workspaceId }
    }

    private func effectiveWorkspaceId(for asset: LibraryAsset) -> String {
        if let workspaceId = asset.workspaceId {
            return workspaceId
        }
        if let runGroupId = asset.runGroupId, let runGroup = runGroupsById[runGroupId] {
            return runGroup.workspaceId
        }
        if let importedWorkspaceId = asset.importedAsset?.workspaceId {
            return importedWorkspaceId
        }
        return defaultWorkspaceId
    }

    private var normalizedQuery: String {
        filters.query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private func matchesFilters(_ asset: LibraryAsset) -> Bool {
        guard filters.selectedFilter.includes(asset) else { return false }
        guard !filters.favoritesOnly || asset.isFavorite else { return false }
        guard filters.selectedModelId == "all" || asset.modelId == filters.selectedModelId else {
            return false
        }
        guard filters.selectedTaskRaw == "all" || asset.task?.rawValue == filters.selectedTaskRaw else {
            return false
        }
        if let selectedCollectionId = filters.selectedCollectionId,
           !asset.collectionIds.contains(selectedCollectionId)
        {
            return false
        }
        guard normalizedQuery.isEmpty || asset.searchableText.contains(normalizedQuery) else {
            return false
        }
        return true
    }

    private func sortComparator(_ lhs: LibraryAsset, _ rhs: LibraryAsset) -> Bool {
        switch filters.selectedSort {
        case .newest:
            if lhs.createdAt != rhs.createdAt {
                return lhs.createdAt > rhs.createdAt
            }
            return lhs.id > rhs.id
        case .lastUsed:
            let lhsLastUsed = lhs.lastUsedAt ?? lhs.createdAt
            let rhsLastUsed = rhs.lastUsedAt ?? rhs.createdAt
            if lhsLastUsed != rhsLastUsed {
                return lhsLastUsed > rhsLastUsed
            }
            return lhs.createdAt > rhs.createdAt
        }
    }

    private func sortTimestamp(for group: LibraryAssetGroupPresentation) -> Date {
        switch filters.selectedSort {
        case .newest:
            return group.updatedAt
        case .lastUsed:
            return group.primaryAsset.lastUsedAt ?? group.updatedAt
        }
    }
}

private extension LibraryAsset {
    var promptHeadline: String? {
        let normalized = prompt
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            return nil
        }
        if normalized.count <= 72 {
            return normalized
        }
        let cutoff = normalized.index(normalized.startIndex, offsetBy: 69)
        let truncated = String(normalized[..<cutoff]).trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(truncated)…"
    }
}
