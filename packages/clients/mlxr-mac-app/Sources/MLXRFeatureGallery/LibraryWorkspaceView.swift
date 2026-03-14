import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI
import UniformTypeIdentifiers

public struct LibraryWorkspaceView: View {
    private let workspaces: [WorkspaceRecord]
    private let activeWorkspaceId: String
    private let focusedWorkspaceId: String?
    private let focusedAssetId: String?
    private let assets: [LibraryAsset]
    private let runGroups: [RunGroupRecord]
    private let collections: [CollectionRecord]
    private let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    private let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    private let onRemoveImportedAsset: @Sendable (String) async -> Void
    private let onSeedComposer: (ComposerSeedRequest) -> Void
    private let onSelectWorkspace: (String) -> Void
    private let onCreateWorkspace: () -> WorkspaceRecord
    private let onSetWorkspaceCover: (String, String) -> Void
    private let onToggleFavorite: (String) -> Void
    private let onToggleCollection: (String, String) -> Void
    private let onCreateCollection: (String) -> Void

    @State private var selectedFilter: LibraryAssetFilter = .all
    @State private var selectedModelId = "all"
    @State private var selectedTaskRaw = "all"
    @State private var selectedSort: AssetSortMode = .newest
    @State private var favoritesOnly = false
    @State private var selectedCollectionId: String?
    @State private var query = ""
    @State private var browserState = LibraryBrowserState()
    @State private var isPickingImports = false
    @State private var newCollectionName = ""
    @State private var gridColumnCount = 1
    @FocusState private var isLibraryFocused: Bool

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
        onCreateCollection: @escaping (String) -> Void
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
    }

    public var body: some View {
        ZStack {
            VStack(spacing: 0) {
                libraryToolbar

                if let selectedWorkspace = presentation.selectedWorkspace,
                   let projectSummary = presentation.projectSummary(id: selectedWorkspace.id) {
                    projectDetail(projectSummary)
                } else {
                    ProjectBrowserView(
                        projects: presentation.projectSummaries,
                        onMaterialize: onMaterialize,
                        onOpenProject: { workspaceId in
                            browserState.openProject(workspaceId)
                        },
                        onCreateProject: createProjectFromBrowser
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let viewerAsset {
                viewerOverlay(for: viewerAsset)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    .zIndex(20)
            }
        }
        .background(AdaptiveBackground())
        .focusable()
        .focused($isLibraryFocused)
        .onAppear {
            isLibraryFocused = true
            applyFocusedRoute()
        }
        .onChange(of: focusedWorkspaceId) { _, _ in
            applyFocusedRoute()
        }
        .onChange(of: focusedAssetId) { _, _ in
            applyFocusedRoute()
        }
        .onChange(of: browserState.selectedWorkspaceId) { _, newValue in
            if let newValue {
                onSelectWorkspace(newValue)
            }
        }
        .onChange(of: presentation.orderedPrimaryAssetIds) { _, visiblePrimaryIds in
            browserState.pruneVisibleState(
                visiblePrimaryAssetIds: visiblePrimaryIds,
                isViewerAssetVisible: presentation.viewerAsset(for: browserState.viewerAssetId) != nil
            )
        }
        .onMoveCommand { direction in
            moveSelection(direction)
        }
        .onExitCommand {
            if browserState.viewerAssetId != nil {
                browserState.closeViewer()
            } else if browserState.selectedPrimaryAssetId != nil {
                browserState.clearAssetSelection()
            } else if browserState.selectedWorkspaceId != nil {
                browserState.clearProjectSelection()
            }
        }
        .onKeyPress(.return) {
            openSelectedAsset()
            return .handled
        }
        .onKeyPress(.space) {
            openSelectedAsset()
            return .handled
        }
        .onDeleteCommand {
            Task {
                await removeSelectedImportedAsset()
            }
        }
        .fileImporter(
            isPresented: $isPickingImports,
            allowedContentTypes: [.image, .movie, .audio],
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result else { return }
            Task {
                let imported = await onImportAssets(urls)
                browserState.handleImportedAssets(imported)
            }
        }
    }

    private var currentFilters: LibraryFilterState {
        LibraryFilterState(
            selectedFilter: selectedFilter,
            selectedModelId: selectedModelId,
            selectedTaskRaw: selectedTaskRaw,
            selectedSort: selectedSort,
            favoritesOnly: favoritesOnly,
            selectedCollectionId: selectedCollectionId,
            query: query
        )
    }

    private var presentation: LibraryPresentationModel {
        LibraryPresentationModel(
            assets: assets,
            workspaces: workspaces,
            runGroups: runGroups,
            collections: collections,
            filters: currentFilters,
            selectedWorkspaceId: browserState.selectedWorkspaceId,
            defaultWorkspaceId: activeWorkspaceId
        )
    }

    private var viewerAsset: LibraryAsset? {
        presentation.viewerAsset(for: browserState.viewerAssetId)
    }

    private var libraryToolbar: some View {
        VStack(spacing: MLXRSpacing.md) {
            HStack(spacing: MLXRSpacing.sm) {
                if let project = presentation.projectSummary(id: browserState.selectedWorkspaceId) {
                    Button {
                        browserState.clearProjectSelection()
                    } label: {
                        Label("Projects", systemImage: "chevron.left")
                    }
                    .buttonStyle(.bordered)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(project.title)
                            .font(MLXRType.titleSmall)
                            .foregroundStyle(MLXRColor.textPrimary)
                        Text("\(project.assetCount) assets across \(project.runGroupCount) sets")
                            .font(MLXRType.captionLarge)
                            .foregroundStyle(MLXRColor.textSecondary)
                    }
                } else {
                    Text("Projects")
                        .font(MLXRType.titleLarge)
                        .foregroundStyle(MLXRColor.textPrimary)
                }

                Spacer(minLength: 0)

                TextField(
                    browserState.selectedWorkspaceId == nil
                        ? "Search projects, prompts, or filenames"
                        : "Search prompts, models, or filenames",
                    text: $query
                )
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 420)

                if browserState.selectedWorkspaceId == nil {
                    Button("New Project", action: createProjectFromBrowser)
                        .buttonStyle(.bordered)
                }

                Button("Import from Finder") {
                    if browserState.selectedWorkspaceId == nil {
                        createProjectFromBrowser()
                    }
                    isPickingImports = true
                }
                .buttonStyle(.borderedProminent)
            }

            if browserState.selectedWorkspaceId != nil {
                detailFiltersBar
            }
        }
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.top, MLXRSpacing.xl)
        .padding(.bottom, MLXRSpacing.md)
    }

    private var detailFiltersBar: some View {
        HStack(spacing: MLXRSpacing.md) {
            Picker("Media", selection: $selectedFilter) {
                ForEach(LibraryAssetFilter.allCases) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 300)

            Toggle(isOn: $favoritesOnly) {
                Text("Favorites")
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
            }
            .toggleStyle(.switch)
            .frame(width: 120)

            Picker("Model", selection: $selectedModelId) {
                Text("All Models").tag("all")
                ForEach(presentation.modelOptions, id: \.self) { modelId in
                    Text(modelId).tag(modelId)
                }
            }
            .frame(maxWidth: 220)

            Picker("Workflow", selection: $selectedTaskRaw) {
                Text("All Workflows").tag("all")
                ForEach(presentation.taskOptions, id: \.rawValue) { task in
                    Text(task.title).tag(task.rawValue)
                }
            }
            .frame(maxWidth: 220)

            Picker("Sort", selection: $selectedSort) {
                ForEach(AssetSortMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 220)

            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private func projectDetail(_ project: ProjectSummaryPresentation) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                ProjectHeaderView(
                    project: project,
                    selectedAsset: selectedPrimaryAsset,
                    onMaterialize: onMaterialize,
                    onSetCover: setCoverAction(for: project)
                )

                if presentation.groups.isEmpty {
                    EmptyStateView(
                        title: "Nothing matches in this project",
                        subtitle: "Adjust the filters, import a reference, or prompt from the composer to keep building this thread.",
                        systemImage: "photo.stack"
                    )
                    .padding(.horizontal, MLXRSpacing.xl)
                    .padding(.vertical, MLXRSpacing.xl)
                } else {
                    LibraryGridView(
                        groups: presentation.groups,
                        selectedPrimaryAssetId: browserState.selectedPrimaryAssetId,
                        onSelect: { assetId in
                            browserState.selectAsset(assetId)
                        },
                        onOpen: { assetId in
                            browserState.openAsset(assetId)
                        },
                        onMaterialize: onMaterialize,
                        onGridMetricsChange: { columns in
                            gridColumnCount = max(columns, 1)
                        }
                    )
                    .frame(minHeight: 420)
                }
            }
            .padding(.bottom, MLXRSpacing.xl)
        }
    }

    private func viewerOverlay(for asset: LibraryAsset) -> some View {
        ZStack {
            Color.black.opacity(0.62)
                .ignoresSafeArea()

            LibraryViewerSheet(
                asset: asset,
                group: presentation.group(containing: asset.id),
                collections: collections,
                onMaterialize: onMaterialize,
                onRemoveImportedAsset: onRemoveImportedAsset,
                onSeedComposer: { request in
                    onSeedComposer(request)
                    selectPrimaryAsset(for: request.focusedAssetId ?? asset.id)
                    browserState.closeViewer()
                },
                onSetProjectCover: {
                    guard let workspaceId = browserState.selectedWorkspaceId else {
                        return
                    }
                    onSetWorkspaceCover(workspaceId, asset.id)
                },
                onToggleFavorite: onToggleFavorite,
                onToggleCollection: onToggleCollection,
                onShowAsset: { nextAsset in
                    browserState.openAsset(nextAsset.id)
                    selectPrimaryAsset(for: nextAsset.id)
                },
                onShowPrevious: previousViewerAction,
                onShowNext: nextViewerAction,
                onClose: {
                    browserState.closeViewer()
                }
            )
            .padding(MLXRSpacing.xl)
        }
    }

    private var previousViewerAction: (() -> Void)? {
        guard let viewerAsset else { return nil }
        let primaryId = presentation.group(containing: viewerAsset.id)?.primaryAsset.id ?? viewerAsset.id
        guard let index = presentation.orderedPrimaryAssetIds.firstIndex(of: primaryId), index > 0 else {
            return nil
        }
        return {
            let previousId = presentation.orderedPrimaryAssetIds[index - 1]
            browserState.openAsset(previousId)
        }
    }

    private var nextViewerAction: (() -> Void)? {
        guard let viewerAsset else { return nil }
        let primaryId = presentation.group(containing: viewerAsset.id)?.primaryAsset.id ?? viewerAsset.id
        guard let index = presentation.orderedPrimaryAssetIds.firstIndex(of: primaryId),
              index < presentation.orderedPrimaryAssetIds.count - 1
        else {
            return nil
        }
        return {
            let nextId = presentation.orderedPrimaryAssetIds[index + 1]
            browserState.openAsset(nextId)
        }
    }

    private func openSelectedAsset() {
        guard browserState.selectedWorkspaceId != nil else { return }
        guard let selectedPrimaryAssetId = browserState.selectedPrimaryAssetId else { return }
        browserState.openAsset(selectedPrimaryAssetId)
    }

    private func selectPrimaryAsset(for assetId: String) {
        browserState.selectAsset(presentation.group(containing: assetId)?.primaryAsset.id ?? assetId)
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        guard browserState.selectedWorkspaceId != nil else { return }
        let orderedPrimaryAssetIds = presentation.orderedPrimaryAssetIds
        guard !orderedPrimaryAssetIds.isEmpty else { return }
        guard let selectedPrimaryAssetId = browserState.selectedPrimaryAssetId,
              let currentIndex = orderedPrimaryAssetIds.firstIndex(of: selectedPrimaryAssetId)
        else {
            if let firstPrimaryAssetId = orderedPrimaryAssetIds.first {
                browserState.selectAsset(firstPrimaryAssetId)
            }
            return
        }

        let offset: Int
        switch direction {
        case .left:
            offset = -1
        case .right:
            offset = 1
        case .up:
            offset = -gridColumnCount
        case .down:
            offset = gridColumnCount
        @unknown default:
            offset = 0
        }

        let nextIndex = min(max(currentIndex + offset, 0), orderedPrimaryAssetIds.count - 1)
        browserState.selectAsset(orderedPrimaryAssetIds[nextIndex])
    }

    private func removeSelectedImportedAsset() async {
        let targetAssetId = browserState.viewerAssetId ?? browserState.selectedPrimaryAssetId
        guard let targetAsset = presentation.viewerAsset(for: targetAssetId) else {
            return
        }
        guard targetAsset.isImported else {
            return
        }
        await onRemoveImportedAsset(targetAsset.id)
        browserState.removeAsset(targetAsset.id)
    }

    private func createProjectFromBrowser() {
        let workspace = onCreateWorkspace()
        browserState.openProject(workspace.id)
    }

    private func applyFocusedRoute() {
        browserState.applyFocusedRoute(
            workspaceId: focusedWorkspaceId,
            assetId: focusedAssetId
        )
    }

    private var selectedPrimaryAsset: LibraryAsset? {
        presentation.viewerAsset(for: browserState.selectedPrimaryAssetId)
    }

    private func setCoverAction(for project: ProjectSummaryPresentation) -> (() -> Void)? {
        guard let workspaceId = browserState.selectedWorkspaceId,
              let selectedPrimaryAsset,
              selectedPrimaryAsset.id != project.heroAsset?.id
        else {
            return nil
        }
        return {
            onSetWorkspaceCover(workspaceId, selectedPrimaryAsset.id)
        }
    }
}
