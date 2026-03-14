import MLXRAppDomain
import SwiftUI

@MainActor
public struct GalleryScreen: View {
    let assets: [LibraryAsset]
    let runGroups: [RunGroupRecord]
    let collections: [CollectionRecord]
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    let onRemoveImportedAsset: @Sendable (String) async -> Void
    let onOpenInStudio: (StudioOpenRequest) -> Void
    let onToggleFavorite: (String) -> Void
    let onToggleCollection: (String, String) -> Void
    let onCreateCollection: (String) -> Void

    public init(
        assets: [LibraryAsset],
        runGroups: [RunGroupRecord],
        collections: [CollectionRecord],
        onMaterialize: @escaping @Sendable (LibraryAsset) async -> URL?,
        onImportAssets: @escaping @Sendable ([URL]) async -> [ImportedAssetRecord],
        onRemoveImportedAsset: @escaping @Sendable (String) async -> Void,
        onOpenInStudio: @escaping (StudioOpenRequest) -> Void,
        onToggleFavorite: @escaping (String) -> Void,
        onToggleCollection: @escaping (String, String) -> Void,
        onCreateCollection: @escaping (String) -> Void
    ) {
        self.assets = assets
        self.runGroups = runGroups
        self.collections = collections
        self.onMaterialize = onMaterialize
        self.onImportAssets = onImportAssets
        self.onRemoveImportedAsset = onRemoveImportedAsset
        self.onOpenInStudio = onOpenInStudio
        self.onToggleFavorite = onToggleFavorite
        self.onToggleCollection = onToggleCollection
        self.onCreateCollection = onCreateCollection
    }

    public var body: some View {
        LibraryWorkspaceView(
            assets: assets,
            runGroups: runGroups,
            collections: collections,
            onMaterialize: onMaterialize,
            onImportAssets: onImportAssets,
            onRemoveImportedAsset: onRemoveImportedAsset,
            onOpenInStudio: onOpenInStudio,
            onToggleFavorite: onToggleFavorite,
            onToggleCollection: onToggleCollection,
            onCreateCollection: onCreateCollection
        )
    }
}
