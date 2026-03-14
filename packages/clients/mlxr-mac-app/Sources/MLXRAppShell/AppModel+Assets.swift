import Foundation
import MLXRAppDomain

@MainActor
extension MLXRAppModel {
    public var libraryAssets: [LibraryAsset] {
        let generated = LibraryAsset.fromGeneratedEntries(libraryEntries, metadataById: assetRecords)
        let imported = LibraryAsset.fromImported(importedAssets, metadataById: assetRecords)
        return (generated + imported).sorted {
            let lhsSort = $0.lastUsedAt ?? $0.createdAt
            let rhsSort = $1.lastUsedAt ?? $1.createdAt
            if lhsSort != rhsSort {
                return lhsSort > rhsSort
            }
            return $0.createdAt > $1.createdAt
        }
    }

    public var recentLibraryAssets: [LibraryAsset] {
        Array(libraryAssets.prefix(12))
    }

    public var latestGeneratedAsset: LibraryAsset? {
        libraryAssets.first(where: \.isGenerated)
    }

    public func importExternalAssets(from urls: [URL]) async -> [ImportedAssetRecord] {
        guard !urls.isEmpty else { return [] }
        do {
            let updated = try importedAssetStore.importFiles(
                at: urls,
                existing: importedAssets,
                workspaceId: activeWorkspaceId
            )
            let existingIds = Set(importedAssets.map { $0.id })
            let createdIds = Set(updated.map { $0.id }).subtracting(existingIds)
            importedAssets = updated
            globalError = nil
            return importedAssets.filter { createdIds.contains($0.id) }
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
}
