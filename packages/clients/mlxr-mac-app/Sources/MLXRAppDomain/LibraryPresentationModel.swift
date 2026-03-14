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
    private let runGroupsById: [String: RunGroupRecord]
    public let collections: [CollectionRecord]
    public let filters: LibraryFilterState

    public init(
        assets: [LibraryAsset],
        runGroups: [RunGroupRecord],
        collections: [CollectionRecord],
        filters: LibraryFilterState
    ) {
        self.assets = assets
        self.runGroupsById = Dictionary(uniqueKeysWithValues: runGroups.map { ($0.id, $0) })
        self.collections = collections
        self.filters = filters
    }

    public var modelOptions: [String] {
        Array(Set(assets.compactMap(\.modelId))).sorted()
    }

    public var taskOptions: [ProductTask] {
        Array(Set(assets.compactMap(\.task))).sorted { $0.title < $1.title }
    }

    public func count(for filter: LibraryAssetFilter) -> Int {
        assets.filter { filter.includes($0) }.count
    }

    public func count(for collectionId: String?) -> Int {
        guard let collectionId else { return assets.count }
        return assets.filter { $0.collectionIds.contains(collectionId) }.count
    }

    public var filteredAssets: [LibraryAsset] {
        let trimmedQuery = filters.query
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return assets.filter { asset in
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
            guard trimmedQuery.isEmpty || asset.searchableText.contains(trimmedQuery) else {
                return false
            }
            return true
        }
        .sorted(by: sortComparator)
    }

    public var groups: [LibraryAssetGroupPresentation] {
        let groupedAssets = Dictionary(grouping: filteredAssets) { asset in
            asset.runGroupId ?? asset.id
        }
        return groupedAssets.compactMap { groupId, assets in
            let sortedAssets = assets.sorted(by: sortComparator)
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
