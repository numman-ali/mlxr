import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI
import UniformTypeIdentifiers

public struct LibraryWorkspaceView: View {
    private let workspaces: [WorkspaceRecord]
    private let defaultWorkspaceId: String
    private let focusedWorkspaceId: String?
    private let focusedAssetId: String?
    private let assets: [LibraryAsset]
    private let runGroups: [RunGroupRecord]
    private let collections: [CollectionRecord]
    private let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    private let onImportAssets: @Sendable ([URL], String) async -> [ImportedAssetRecord]
    private let onRemoveImportedAsset: @Sendable (String) async -> Void
    private let onSeedComposer: (ComposerSeedRequest) -> Void
    private let onSelectWorkspace: (String) -> Void
    private let onShowProjectBrowser: () -> Void
    private let onCreateWorkspace: () -> WorkspaceRecord
    private let onSetWorkspaceCover: (String, String) -> Void
    private let onToggleFavorite: (String) -> Void
    private let onToggleCollection: (String, String) -> Void
    private let onCreateCollection: (String) -> Void
    private let onViewerPresentationChange: (Bool) -> Void

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
    @State private var projectGridColumnCount = 1
    @State private var browserProjectId: String?
    @FocusState private var isLibraryFocused: Bool
    @Environment(\.mlxrBottomOverlayInset) private var bottomOverlayInset

    public init(
        workspaces: [WorkspaceRecord],
        defaultWorkspaceId: String,
        focusedWorkspaceId: String?,
        focusedAssetId: String?,
        assets: [LibraryAsset],
        runGroups: [RunGroupRecord],
        collections: [CollectionRecord],
        onMaterialize: @escaping @Sendable (LibraryAsset) async -> URL?,
        onImportAssets: @escaping @Sendable ([URL], String) async -> [ImportedAssetRecord],
        onRemoveImportedAsset: @escaping @Sendable (String) async -> Void,
        onSeedComposer: @escaping (ComposerSeedRequest) -> Void,
        onSelectWorkspace: @escaping (String) -> Void,
        onShowProjectBrowser: @escaping () -> Void,
        onCreateWorkspace: @escaping () -> WorkspaceRecord,
        onSetWorkspaceCover: @escaping (String, String) -> Void,
        onToggleFavorite: @escaping (String) -> Void,
        onToggleCollection: @escaping (String, String) -> Void,
        onCreateCollection: @escaping (String) -> Void,
        onViewerPresentationChange: @escaping (Bool) -> Void
    ) {
        self.workspaces = workspaces
        self.defaultWorkspaceId = defaultWorkspaceId
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
        self.onShowProjectBrowser = onShowProjectBrowser
        self.onCreateWorkspace = onCreateWorkspace
        self.onSetWorkspaceCover = onSetWorkspaceCover
        self.onToggleFavorite = onToggleFavorite
        self.onToggleCollection = onToggleCollection
        self.onCreateCollection = onCreateCollection
        self.onViewerPresentationChange = onViewerPresentationChange
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
                        selectedProjectId: browserProjectId,
                        onMaterialize: onMaterialize,
                        onOpenProject: { workspaceId in
                            browserProjectId = workspaceId
                            browserState.openProject(workspaceId)
                        },
                        onGridMetricsChange: { columns in
                            projectGridColumnCount = max(columns, 1)
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
            ensureBrowserProjectSelection()
            onViewerPresentationChange(viewerAsset != nil)
        }
        .onChange(of: focusedWorkspaceId) { _, _ in
            applyFocusedRoute()
        }
        .onChange(of: focusedAssetId) { _, _ in
            applyFocusedRoute()
        }
        .onChange(of: browserState.selectedWorkspaceId) { _, newValue in
            if let newValue {
                browserProjectId = newValue
                onSelectWorkspace(newValue)
            } else {
                resetProjectOnlyFilters()
                onShowProjectBrowser()
                ensureBrowserProjectSelection()
            }
        }
        .onChange(of: presentation.projectSummaries.map(\.id)) { _, _ in
            ensureBrowserProjectSelection()
        }
        .onChange(of: presentation.visibleAssetIds) { _, _ in
            browserState.pruneVisibleState(
                visiblePrimaryAssetIds: presentation.orderedPrimaryAssetIds,
                isViewerAssetVisible: presentation.visibleViewerAsset(for: browserState.viewerAssetId) != nil
            )
        }
        .onChange(of: browserState.viewerAssetId) { _, newValue in
            onViewerPresentationChange(newValue != nil)
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
                showProjectBrowser()
            }
        }
        .onKeyPress(.return) {
            openSelectedAsset() ? .handled : .ignored
        }
        .onKeyPress(.space) {
            openSelectedAsset() ? .handled : .ignored
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
                let targetWorkspaceId = browserState.selectedWorkspaceId ?? WorkspaceRecord(title: "New Project").id
                let imported = await onImportAssets(urls, targetWorkspaceId)
                guard !imported.isEmpty else { return }
                browserState.handleImportedAssets(imported)
            }
        }
        .onDisappear {
            onViewerPresentationChange(false)
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
            defaultWorkspaceId: defaultWorkspaceId
        )
    }

    private var viewerAsset: LibraryAsset? {
        presentation.viewerAsset(for: browserState.viewerAssetId)
    }

    private var libraryToolbar: some View {
        VStack(spacing: MLXRSpacing.sm) {
            HStack(spacing: MLXRSpacing.sm) {
                if let project = presentation.projectSummary(id: browserState.selectedWorkspaceId) {
                    Button {
                        showProjectBrowser()
                    } label: {
                        Label("Projects", systemImage: "chevron.left")
                    }
                    .buttonStyle(.bordered)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(project.title)
                            .font(MLXRType.titleSmall)
                            .foregroundStyle(MLXRColor.textPrimary)
                        Text("\(project.assetCount) assets across \(project.runGroupCount) sets")
                            .font(MLXRType.captionSmall)
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
                    isPickingImports = true
                }
                .buttonStyle(.borderedProminent)
            }

            detailFiltersBarContainer
        }
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.top, MLXRSpacing.lg)
        .padding(.bottom, MLXRSpacing.sm)
    }

    private var detailFiltersBarContainer: some View {
        Group {
            if browserState.selectedWorkspaceId != nil {
                detailFiltersBar
            } else {
                browserFiltersBar
            }
        }
        .frame(maxWidth: .infinity, minHeight: 36, maxHeight: 36, alignment: .leading)
    }

    private var browserFiltersBar: some View {
        HStack(spacing: MLXRSpacing.md) {
            Picker("Media", selection: $selectedFilter) {
                ForEach(LibraryAssetFilter.allCases) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 300)

            Toggle(isOn: $favoritesOnly) {
                Text("Favorites")
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
            }
            .toggleStyle(.switch)
            .frame(width: 120)

            Spacer(minLength: 0)
        }
    }

    private var detailFiltersBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: MLXRSpacing.md) {
                Picker("Media", selection: $selectedFilter) {
                    ForEach(LibraryAssetFilter.allCases) { filter in
                        Text(filter.rawValue).tag(filter)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)

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
                .frame(width: 220)

                Picker("Workflow", selection: $selectedTaskRaw) {
                    Text("All Workflows").tag("all")
                    ForEach(presentation.taskOptions, id: \.rawValue) { task in
                        Text(task.title).tag(task.rawValue)
                    }
                }
                .frame(width: 220)

                Picker("Sort", selection: $selectedSort) {
                    ForEach(AssetSortMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)

                Spacer(minLength: 0)
            }
        }
    }

    @ViewBuilder
    private func projectDetail(_ project: ProjectSummaryPresentation) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            ProjectHeaderView(
                project: project,
                selectedAsset: selectedPrimaryAsset,
                onMaterialize: onMaterialize,
                onSetCover: setCoverAction(for: project)
            )

            if !pendingRunGroups.isEmpty {
                pendingRunGroupsStrip
                    .padding(.horizontal, MLXRSpacing.lg)
            }

            if presentation.groups.isEmpty {
                EmptyStateView(
                    title: "Nothing matches in this project",
                    subtitle: "Adjust the filters, import a reference, or prompt from the composer to keep building this thread.",
                    systemImage: "photo.stack"
                )
                .padding(.horizontal, MLXRSpacing.xl)
                .padding(.top, MLXRSpacing.xl)
                .padding(.bottom, max(MLXRSpacing.xl, bottomOverlayInset))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func viewerOverlay(for asset: LibraryAsset) -> some View {
        GeometryReader { proxy in
            let horizontalInset = min(max(proxy.size.width * 0.03, 12), MLXRSpacing.xl)
            let verticalInset = min(max(proxy.size.height * 0.03, 12), MLXRSpacing.xl)
            let sheetWidth = min(max(proxy.size.width - (horizontalInset * 2), 240), 1_360)
            let sheetHeight = min(max(proxy.size.height - (verticalInset * 2), 220), 880)
            ZStack {
                Color.black.opacity(0.74)
                    .ignoresSafeArea()

                LibraryViewerSheet(
                    asset: asset,
                    group: presentation.group(containing: asset.id),
                    collections: collections,
                    onMaterialize: onMaterialize,
                    onRemoveImportedAsset: onRemoveImportedAsset,
                    onSeedComposer: { request in
                        onSeedComposer(request)
                        if let workspaceId = request.workspaceId ?? browserState.selectedWorkspaceId {
                            browserProjectId = workspaceId
                            browserState.openProject(workspaceId)
                        }
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
                .frame(
                    width: sheetWidth,
                    height: sheetHeight
                )
                .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.3), radius: 28, y: 14)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
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

    private func openSelectedAsset() -> Bool {
        guard browserState.selectedWorkspaceId != nil else {
            let targetProjectId = browserProjectId ?? presentation.projectSummaries.first?.id
            guard let targetProjectId else { return false }
            browserProjectId = targetProjectId
            browserState.openProject(targetProjectId)
            return true
        }
        guard let selectedPrimaryAssetId = browserState.selectedPrimaryAssetId else { return false }
        browserState.openAsset(selectedPrimaryAssetId)
        return true
    }

    private func selectPrimaryAsset(for assetId: String) {
        browserState.selectAsset(presentation.group(containing: assetId)?.primaryAsset.id ?? assetId)
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        guard browserState.selectedWorkspaceId != nil else {
            let orderedProjectIds = presentation.projectSummaries.map(\.id)
            guard !orderedProjectIds.isEmpty else { return }
            guard let browserProjectId,
                  let currentIndex = orderedProjectIds.firstIndex(of: browserProjectId)
            else {
                self.browserProjectId = orderedProjectIds.first
                return
            }

            let offset: Int
            switch direction {
            case .left:
                offset = -1
            case .right:
                offset = 1
            case .up:
                offset = -projectGridColumnCount
            case .down:
                offset = projectGridColumnCount
            @unknown default:
                offset = 0
            }

            let nextIndex = min(max(currentIndex + offset, 0), orderedProjectIds.count - 1)
            self.browserProjectId = orderedProjectIds[nextIndex]
            return
        }
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
        browserProjectId = workspace.id
        browserState.openProject(workspace.id)
        onSelectWorkspace(workspace.id)
    }

    private func showProjectBrowser() {
        resetProjectOnlyFilters()
        browserProjectId = browserState.selectedWorkspaceId ?? browserProjectId
        browserState.clearProjectSelection()
    }

    private func resetProjectOnlyFilters() {
        var filters = currentFilters
        filters.clearProjectScopedSelections()
        selectedModelId = filters.selectedModelId
        selectedTaskRaw = filters.selectedTaskRaw
        selectedCollectionId = filters.selectedCollectionId
    }

    private func applyFocusedRoute() {
        browserState.applyFocusedRoute(
            workspaceId: focusedWorkspaceId,
            assetId: focusedAssetId
        )
        if let focusedWorkspaceId {
            browserProjectId = focusedWorkspaceId
        }
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

    private func ensureBrowserProjectSelection() {
        guard browserState.selectedWorkspaceId == nil else { return }
        let projectIds = presentation.projectSummaries.map(\.id)
        guard !projectIds.isEmpty else {
            browserProjectId = nil
            return
        }
        if let browserProjectId, projectIds.contains(browserProjectId) {
            return
        }
        browserProjectId = projectIds.first
    }

    private var pendingRunGroups: [RunGroupRecord] {
        guard let workspaceId = browserState.selectedWorkspaceId else {
            return []
        }
        let visibleRunGroupIds = Set(presentation.groups.compactMap(\.runGroupId))
        return runGroups
            .filter { $0.workspaceId == workspaceId }
            .filter { $0.state == .queued || $0.state == .running || $0.state == .failed }
            .filter { visibleRunGroupIds.contains($0.id) == false || $0.assetIds.isEmpty }
            .sorted { lhs, rhs in
                if lhs.updatedAt != rhs.updatedAt {
                    return lhs.updatedAt > rhs.updatedAt
                }
                return lhs.createdAt > rhs.createdAt
            }
    }

    private var pendingRunGroupsStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: MLXRSpacing.sm) {
                ForEach(pendingRunGroups) { runGroup in
                    PendingRunGroupCard(runGroup: runGroup)
                }
            }
        }
    }
}

private struct PendingRunGroupCard: View {
    let runGroup: RunGroupRecord

    var body: some View {
        HStack(alignment: .top, spacing: MLXRSpacing.sm) {
            VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                Text(runGroup.title)
                    .font(MLXRType.bodySmall)
                    .fontWeight(.semibold)
                    .foregroundStyle(MLXRColor.textPrimary)
                    .lineLimit(1)

                Text(runGroup.task.title)
                    .font(MLXRType.captionLarge)
                    .foregroundStyle(MLXRColor.textSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: MLXRSpacing.md)

            VStack(alignment: .trailing, spacing: MLXRSpacing.xxs) {
                StatusPill(label: stateLabel, tint: stateTint)
                if runGroup.variationCount > 1 {
                    Text("\(runGroup.variationCount) outputs")
                        .font(MLXRType.captionSmall)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
            }
        }
        .padding(.horizontal, MLXRSpacing.md)
        .padding(.vertical, MLXRSpacing.sm)
        .frame(width: 280, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                .fill(MLXRColor.surfaceCard)
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
        )
    }

    private var stateLabel: String {
        switch runGroup.state {
        case .queued:
            "Queued"
        case .running:
            "Generating"
        case .failed:
            "Failed"
        case .completed:
            "Completed"
        case .cancelled:
            "Cancelled"
        }
    }

    private var stateTint: Color {
        switch runGroup.state {
        case .queued:
            MLXRColor.brandWarm
        case .running:
            MLXRColor.brandPrimary
        case .failed:
            MLXRColor.brandDanger
        case .completed:
            MLXRColor.brandSecondary
        case .cancelled:
            MLXRColor.textTertiary
        }
    }
}
