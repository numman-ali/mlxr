public struct LibraryBrowserState: Equatable, Sendable {
    public var selectedWorkspaceId: String?
    public var selectedPrimaryAssetId: String?
    public var viewerAssetId: String?

    public init(
        selectedWorkspaceId: String? = nil,
        selectedPrimaryAssetId: String? = nil,
        viewerAssetId: String? = nil
    ) {
        self.selectedWorkspaceId = selectedWorkspaceId
        self.selectedPrimaryAssetId = selectedPrimaryAssetId
        self.viewerAssetId = viewerAssetId
    }

    public mutating func openProject(_ workspaceId: String) {
        selectedWorkspaceId = workspaceId
        selectedPrimaryAssetId = nil
        viewerAssetId = nil
    }

    public mutating func clearProjectSelection() {
        selectedWorkspaceId = nil
        selectedPrimaryAssetId = nil
        viewerAssetId = nil
    }

    public mutating func selectAsset(_ assetId: String) {
        selectedPrimaryAssetId = assetId
    }

    public mutating func openAsset(_ assetId: String) {
        selectedPrimaryAssetId = assetId
        viewerAssetId = assetId
    }

    public mutating func closeViewer() {
        viewerAssetId = nil
    }

    public mutating func clearAssetSelection() {
        selectedPrimaryAssetId = nil
        viewerAssetId = nil
    }

    public mutating func applyFocusedRoute(
        workspaceId: String?,
        assetId: String?
    ) {
        switch (workspaceId, assetId) {
        case let (.some(workspaceId), .some(assetId)):
            selectedWorkspaceId = workspaceId
            selectedPrimaryAssetId = assetId
            viewerAssetId = assetId
        case let (.some(workspaceId), .none):
            openProject(workspaceId)
        case let (.none, .some(assetId)):
            selectedWorkspaceId = nil
            selectedPrimaryAssetId = assetId
            viewerAssetId = assetId
        case (.none, .none):
            clearProjectSelection()
        }
    }

    public mutating func handleImportedAssets(_ importedAssets: [ImportedAssetRecord]) {
        guard let first = importedAssets.first else {
            return
        }
        if let workspaceId = first.workspaceId {
            selectedWorkspaceId = workspaceId
        }
        selectedPrimaryAssetId = first.id
        viewerAssetId = first.id
    }

    public mutating func pruneVisibleState(
        visiblePrimaryAssetIds: [String],
        isViewerAssetVisible: Bool
    ) {
        if let selectedPrimaryAssetId, !visiblePrimaryAssetIds.contains(selectedPrimaryAssetId) {
            self.selectedPrimaryAssetId = nil
        }
        if !isViewerAssetVisible {
            viewerAssetId = nil
        }
    }

    public mutating func removeAsset(_ assetId: String) {
        if viewerAssetId == assetId {
            viewerAssetId = nil
        }
        if selectedPrimaryAssetId == assetId {
            selectedPrimaryAssetId = nil
        }
    }
}
