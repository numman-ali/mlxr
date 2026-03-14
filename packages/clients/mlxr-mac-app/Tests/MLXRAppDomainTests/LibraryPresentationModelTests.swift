import Foundation
import Testing
@testable import MLXRAppDomain

@Test
func projectSummariesGroupAssetsByWorkspaceAndScopeSelectedProject() {
    let now = Date(timeIntervalSince1970: 1_000)
    let workspaces = [
        WorkspaceRecord(id: "workspace-a", title: "Moodboard", createdAt: now, lastOpenedAt: now),
        WorkspaceRecord(id: "workspace-b", title: "Ad Concepts", createdAt: now, lastOpenedAt: now.addingTimeInterval(60)),
    ]
    let runGroups = [
        RunGroupRecord(
            id: "group-a",
            workspaceId: "workspace-a",
            task: .imageGenerate,
            title: "Set A",
            jobIds: ["job-a"],
            assetIds: ["asset-a1", "asset-a2"],
            variationCount: 2,
            createdAt: now,
            updatedAt: now.addingTimeInterval(30),
            state: .completed
        ),
        RunGroupRecord(
            id: "group-b",
            workspaceId: "workspace-b",
            task: .videoGenerate,
            title: "Set B",
            jobIds: ["job-b"],
            assetIds: ["asset-b1"],
            variationCount: 1,
            createdAt: now,
            updatedAt: now.addingTimeInterval(90),
            state: .running
        ),
    ]
    let assets = [
        libraryAsset(
            id: "asset-a1",
            title: "First still",
            prompt: "A red umbrella in neon rain",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now,
            runGroupId: "group-a",
            workspaceId: "workspace-a"
        ),
        libraryAsset(
            id: "asset-a2",
            title: "Edited still",
            prompt: "Refine the umbrella lighting",
            modelId: "qwen-image-local",
            task: .imageEdit,
            createdAt: now.addingTimeInterval(10),
            runGroupId: "group-a",
            workspaceId: "workspace-a"
        ),
        libraryAsset(
            id: "asset-b1",
            kind: .video,
            title: "Promo clip",
            prompt: "A mechanical whale gliding above a sunset market",
            modelId: "ltx-2.3-fast-local",
            task: .videoGenerate,
            createdAt: now.addingTimeInterval(70),
            runGroupId: "group-b",
            workspaceId: "workspace-b"
        ),
    ]

    let presentation = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: runGroups,
        collections: [],
        filters: LibraryFilterState(),
        selectedWorkspaceId: "workspace-a",
        defaultWorkspaceId: "workspace-a"
    )

    #expect(presentation.projectSummaries.map { $0.id } == ["workspace-b", "workspace-a"])
    #expect(presentation.selectedWorkspace?.title == "Moodboard")
    #expect(presentation.filteredAssets.map { $0.id } == ["asset-a2", "asset-a1"])
    #expect(presentation.groups.count == 1)
    #expect(presentation.projectSummary(id: "workspace-a")?.runGroupCount == 1)
    #expect(presentation.projectSummary(id: "workspace-b")?.videoCount == 1)
}

@Test
func projectSummariesHonorSearchAcrossProjectTitlesAndAssetPrompts() {
    let now = Date(timeIntervalSince1970: 2_000)
    let workspaces = [
        WorkspaceRecord(id: "workspace-a", title: "Neon Alley", createdAt: now, lastOpenedAt: now),
        WorkspaceRecord(id: "workspace-b", title: "Whale Concepts", createdAt: now, lastOpenedAt: now),
    ]
    let assets = [
        libraryAsset(
            id: "asset-a1",
            title: "Alley still",
            prompt: "A violinist under neon rain",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now,
            workspaceId: "workspace-a"
        ),
        libraryAsset(
            id: "asset-b1",
            title: "Whale frame",
            prompt: "A colossal whale crossing the sunset skyline",
            modelId: "ltx-2.3-fast-local",
            task: .videoGenerate,
            createdAt: now.addingTimeInterval(30),
            workspaceId: "workspace-b"
        ),
    ]

    let titleSearch = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: [],
        collections: [],
        filters: .init(query: "neon"),
        selectedWorkspaceId: nil,
        defaultWorkspaceId: "workspace-a"
    )

    let promptSearch = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: [],
        collections: [],
        filters: .init(query: "sunset"),
        selectedWorkspaceId: nil,
        defaultWorkspaceId: "workspace-a"
    )

    #expect(titleSearch.projectSummaries.map(\.id) == ["workspace-a"])
    #expect(promptSearch.projectSummaries.map(\.id) == ["workspace-b"])
}

@Test
func workspaceResolutionFallsBackToRunGroupImportedRecordAndDefaultWorkspace() {
    let now = Date(timeIntervalSince1970: 3_000)
    let workspaces = [
        WorkspaceRecord(id: "workspace-a", title: "Fallback A", createdAt: now, lastOpenedAt: now),
        WorkspaceRecord(id: "workspace-b", title: "Fallback B", createdAt: now, lastOpenedAt: now),
    ]
    let runGroups = [
        RunGroupRecord(
            id: "group-b",
            workspaceId: "workspace-b",
            task: .imageGenerate,
            title: "Recovered set",
            jobIds: ["job-b"],
            assetIds: ["group-derived"],
            variationCount: 1,
            createdAt: now,
            updatedAt: now.addingTimeInterval(10),
            state: .completed
        ),
    ]
    let imported = ImportedAssetRecord(
        id: "imported-asset",
        workspaceId: "workspace-b",
        title: "Imported board",
        sourcePath: "/tmp/imported.png",
        storageKey: "imports/imported.png",
        mediaType: "image/png",
        kind: .image,
        importedAt: now
    )
    let assets = [
        libraryAsset(
            id: "group-derived",
            title: "Recovered still",
            prompt: "Recovered prompt",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now,
            runGroupId: "group-b",
            workspaceId: nil
        ),
        libraryAsset(
            id: "imported-local",
            origin: .imported,
            title: "Imported board",
            prompt: "",
            modelId: nil,
            task: nil,
            createdAt: now.addingTimeInterval(5),
            runGroupId: nil,
            workspaceId: nil,
            importedAsset: imported
        ),
        libraryAsset(
            id: "default-owned",
            title: "Loose asset",
            prompt: "No workspace metadata",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now.addingTimeInterval(8),
            runGroupId: nil,
            workspaceId: nil
        ),
    ]

    let selectedWorkspaceB = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: runGroups,
        collections: [],
        filters: LibraryFilterState(),
        selectedWorkspaceId: "workspace-b",
        defaultWorkspaceId: "workspace-a"
    )
    let selectedWorkspaceA = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: runGroups,
        collections: [],
        filters: LibraryFilterState(),
        selectedWorkspaceId: "workspace-a",
        defaultWorkspaceId: "workspace-a"
    )

    #expect(Set(selectedWorkspaceB.filteredAssets.map { $0.id }) == ["group-derived", "imported-local"])
    #expect(selectedWorkspaceA.filteredAssets.map { $0.id } == ["default-owned"])
}

@Test
func projectSummaryPrefersExplicitWorkspaceCoverAsset() {
    let now = Date(timeIntervalSince1970: 4_000)
    let workspaces = [
        WorkspaceRecord(
            id: "workspace-a",
            title: "Cover Test",
            coverAssetId: "asset-older",
            createdAt: now,
            lastOpenedAt: now
        )
    ]
    let assets = [
        libraryAsset(
            id: "asset-newer",
            title: "Latest frame",
            prompt: "A dramatic castle at dawn",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now.addingTimeInterval(30),
            workspaceId: "workspace-a"
        ),
        libraryAsset(
            id: "asset-older",
            title: "Chosen cover",
            prompt: "An ogre attacking a castle",
            modelId: "z-image-turbo-local",
            task: .imageGenerate,
            createdAt: now,
            workspaceId: "workspace-a"
        ),
    ]

    let presentation = LibraryPresentationModel(
        assets: assets,
        workspaces: workspaces,
        runGroups: [],
        collections: [],
        filters: LibraryFilterState(),
        selectedWorkspaceId: "workspace-a",
        defaultWorkspaceId: "workspace-a"
    )

    #expect(presentation.projectSummary(id: "workspace-a")?.heroAsset?.id == "asset-older")
    #expect(presentation.projectSummary(id: "workspace-a")?.subtitle == "A dramatic castle at dawn")
}

private func libraryAsset(
    id: String,
    origin: LibraryAssetOrigin = .generated,
    kind: LibraryAssetKind = .image,
    title: String,
    prompt: String,
    modelId: String?,
    task: ProductTask?,
    createdAt: Date,
    runGroupId: String? = nil,
    workspaceId: String? = nil,
    importedAsset: ImportedAssetRecord? = nil
) -> LibraryAsset {
    LibraryAsset(
        id: id,
        origin: origin,
        kind: kind,
        title: title,
        prompt: prompt,
        modelId: modelId,
        task: task,
        createdAt: createdAt,
        mediaType: kind == .video ? "video/mp4" : "image/png",
        jobId: nil,
        artifact: nil,
        job: nil,
        importedAsset: importedAsset,
        runGroupId: runGroupId,
        workspaceId: workspaceId
    )
}
