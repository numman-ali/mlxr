import MLXRAppDomain
import SwiftUI

@MainActor
public struct GalleryScreen: View {
    let workspaces: [WorkspaceRecord]
    let activeWorkspaceId: String
    let focusedWorkspaceId: String?
    let focusedAssetId: String?
    let assets: [LibraryAsset]
    let runGroups: [RunGroupRecord]
    let collections: [CollectionRecord]
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    let onRemoveImportedAsset: @Sendable (String) async -> Void
    let onSeedComposer: (ComposerSeedRequest) -> Void
    let onSelectWorkspace: (String) -> Void
    let onCreateWorkspace: () -> WorkspaceRecord
    let onSetWorkspaceCover: (String, String) -> Void
    let onToggleFavorite: (String) -> Void
    let onToggleCollection: (String, String) -> Void
    let onCreateCollection: (String) -> Void
    let onViewerPresentationChange: (Bool) -> Void

    public init(
        workspaces: [WorkspaceRecord],
        activeWorkspaceId: String,
        focusedWorkspaceId: String?,
        focusedAssetId: String?,
        assets: [LibraryAsset],
        runGroups: [RunGroupRecord],
        collections: [CollectionRecord],
        onMaterialize: @escaping @Sendable (LibraryAsset) async -> URL?,
        onImportAssets: @escaping @Sendable ([URL]) async -> [ImportedAssetRecord],
        onRemoveImportedAsset: @escaping @Sendable (String) async -> Void,
        onSeedComposer: @escaping (ComposerSeedRequest) -> Void,
        onSelectWorkspace: @escaping (String) -> Void,
        onCreateWorkspace: @escaping () -> WorkspaceRecord,
        onSetWorkspaceCover: @escaping (String, String) -> Void,
        onToggleFavorite: @escaping (String) -> Void,
        onToggleCollection: @escaping (String, String) -> Void,
        onCreateCollection: @escaping (String) -> Void,
        onViewerPresentationChange: @escaping (Bool) -> Void
    ) {
        self.workspaces = workspaces
        self.activeWorkspaceId = activeWorkspaceId
        self.focusedWorkspaceId = focusedWorkspaceId
        self.focusedAssetId = focusedAssetId
        self.assets = assets
        self.runGroups = runGroups
        self.collections = collections
        self.onMaterialize = onMaterialize
        self.onImportAssets = onImportAssets
        self.onRemoveImportedAsset = onRemoveImportedAsset
        self.onSeedComposer = onSeedComposer
        self.onSelectWorkspace = onSelectWorkspace
        self.onCreateWorkspace = onCreateWorkspace
        self.onSetWorkspaceCover = onSetWorkspaceCover
        self.onToggleFavorite = onToggleFavorite
        self.onToggleCollection = onToggleCollection
        self.onCreateCollection = onCreateCollection
        self.onViewerPresentationChange = onViewerPresentationChange
    }

    public var body: some View {
        LibraryWorkspaceView(
            workspaces: workspaces,
            activeWorkspaceId: activeWorkspaceId,
            focusedWorkspaceId: focusedWorkspaceId,
            focusedAssetId: focusedAssetId,
            assets: assets,
            runGroups: runGroups,
            collections: collections,
            onMaterialize: onMaterialize,
            onImportAssets: onImportAssets,
            onRemoveImportedAsset: onRemoveImportedAsset,
            onSeedComposer: onSeedComposer,
            onSelectWorkspace: onSelectWorkspace,
            onCreateWorkspace: onCreateWorkspace,
            onSetWorkspaceCover: onSetWorkspaceCover,
            onToggleFavorite: onToggleFavorite,
            onToggleCollection: onToggleCollection,
            onCreateCollection: onCreateCollection,
            onViewerPresentationChange: onViewerPresentationChange
        )
    }
}
