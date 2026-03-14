import Foundation
import Testing
@testable import MLXRAppDomain

@Test
func openingProjectResetsAssetAndViewerState() {
    var state = LibraryBrowserState(
        selectedWorkspaceId: "workspace-a",
        selectedPrimaryAssetId: "asset-a",
        viewerAssetId: "asset-a"
    )

    state.openProject("workspace-b")

    #expect(state.selectedWorkspaceId == "workspace-b")
    #expect(state.selectedPrimaryAssetId == nil)
    #expect(state.viewerAssetId == nil)
}

@Test
func openingAssetSelectsItAndShowsViewer() {
    var state = LibraryBrowserState()

    state.openAsset("asset-a")

    #expect(state.selectedPrimaryAssetId == "asset-a")
    #expect(state.viewerAssetId == "asset-a")
}

@Test
func focusedRouteRestoresWorkspaceAndViewer() {
    var state = LibraryBrowserState()

    state.applyFocusedRoute(workspaceId: "workspace-a", assetId: "asset-a")

    #expect(state.selectedWorkspaceId == "workspace-a")
    #expect(state.selectedPrimaryAssetId == "asset-a")
    #expect(state.viewerAssetId == "asset-a")
}

@Test
func importedAssetsOpenWorkspaceAndViewer() {
    let imported = ImportedAssetRecord(
        id: "imported-1",
        workspaceId: "workspace-a",
        title: "Reference",
        sourcePath: "/tmp/reference.png",
        storageKey: "imports/reference.png",
        mediaType: "image/png",
        kind: .image,
        importedAt: Date(timeIntervalSince1970: 1_000)
    )
    var state = LibraryBrowserState()

    state.handleImportedAssets([imported])

    #expect(state.selectedWorkspaceId == "workspace-a")
    #expect(state.selectedPrimaryAssetId == "imported-1")
    #expect(state.viewerAssetId == "imported-1")
}

@Test
func removingSelectedAssetClearsSelectionAndViewer() {
    var state = LibraryBrowserState(
        selectedWorkspaceId: "workspace-a",
        selectedPrimaryAssetId: "asset-a",
        viewerAssetId: "asset-a"
    )

    state.removeAsset("asset-a")

    #expect(state.selectedWorkspaceId == "workspace-a")
    #expect(state.selectedPrimaryAssetId == nil)
    #expect(state.viewerAssetId == nil)
}
