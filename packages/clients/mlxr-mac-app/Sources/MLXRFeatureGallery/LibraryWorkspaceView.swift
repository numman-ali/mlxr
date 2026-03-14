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
    @State private var selectedWorkspaceId: String?
    @State private var selectedPrimaryAssetId: String?
    @State private var viewerAssetId: String?
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
                            selectedWorkspaceId = workspaceId
                            selectedPrimaryAssetId = nil
                            viewerAssetId = nil
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
        .onChange(of: selectedWorkspaceId) { _, newValue in
            if let newValue {
                onSelectWorkspace(newValue)
            }
            selectedPrimaryAssetId = nil
            viewerAssetId = nil
        }
        .onChange(of: presentation.orderedPrimaryAssetIds) { _, visiblePrimaryIds in
            if let selectedPrimaryAssetId, !visiblePrimaryIds.contains(selectedPrimaryAssetId) {
                self.selectedPrimaryAssetId = visiblePrimaryIds.first
            }
            if let viewerAssetId, presentation.viewerAsset(for: viewerAssetId) == nil {
                self.viewerAssetId = nil
            }
        }
        .onMoveCommand { direction in
            moveSelection(direction)
        }
        .onExitCommand {
            if viewerAssetId != nil {
                viewerAssetId = nil
            } else if selectedPrimaryAssetId != nil {
                selectedPrimaryAssetId = nil
            } else if selectedWorkspaceId != nil {
                selectedWorkspaceId = nil
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
                if let first = imported.first {
                    if let workspaceId = first.workspaceId {
                        selectedWorkspaceId = workspaceId
                    }
                    selectedPrimaryAssetId = first.id
                    viewerAssetId = first.id
                }
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
            selectedWorkspaceId: selectedWorkspaceId,
            defaultWorkspaceId: activeWorkspaceId
        )
    }

    private var viewerAsset: LibraryAsset? {
        presentation.viewerAsset(for: viewerAssetId)
    }

    private var libraryToolbar: some View {
        VStack(spacing: MLXRSpacing.md) {
            HStack(spacing: MLXRSpacing.sm) {
                if let project = presentation.projectSummary(id: selectedWorkspaceId) {
                    Button {
                        selectedWorkspaceId = nil
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
                    selectedWorkspaceId == nil
                        ? "Search projects, prompts, or filenames"
                        : "Search prompts, models, or filenames",
                    text: $query
                )
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 420)

                if selectedWorkspaceId == nil {
                    Button("New Project", action: createProjectFromBrowser)
                        .buttonStyle(.bordered)
                }

                Button("Import from Finder") {
                    if selectedWorkspaceId == nil {
                        createProjectFromBrowser()
                    }
                    isPickingImports = true
                }
                .buttonStyle(.borderedProminent)
            }

            if selectedWorkspaceId != nil {
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
                projectHeader(project)

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
                        selectedPrimaryAssetId: selectedPrimaryAssetId,
                        onSelect: { assetId in
                            selectedPrimaryAssetId = assetId
                        },
                        onOpen: { assetId in
                            selectedPrimaryAssetId = assetId
                            viewerAssetId = assetId
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

    private func projectHeader(_ project: ProjectSummaryPresentation) -> some View {
        GlassCard(
            title: project.title,
            subtitle: "Everything created or imported in this project stays together here."
        ) {
            HStack(alignment: .top, spacing: MLXRSpacing.lg) {
                if let heroAsset = project.heroAsset {
                    LibraryAssetThumbnailView(asset: heroAsset, onMaterialize: onMaterialize)
                        .frame(width: 260, height: 180)
                        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous))
                } else {
                    RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                        .fill(MLXRColor.surfaceCard)
                        .frame(width: 260, height: 180)
                        .overlay {
                            VStack(spacing: MLXRSpacing.xs) {
                                Image(systemName: "sparkles.rectangle.stack")
                                    .font(.system(size: 28, weight: .semibold))
                                    .foregroundStyle(MLXRColor.brandPrimary)
                                Text("Create the first result")
                                    .font(MLXRType.captionLarge)
                                    .foregroundStyle(MLXRColor.textSecondary)
                            }
                        }
                }

                VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                    HStack(spacing: MLXRSpacing.xs) {
                        StatusPill(label: "\(project.assetCount) assets", tint: MLXRColor.brandSecondary)
                        StatusPill(label: "\(project.runGroupCount) sets", tint: MLXRColor.brandWarm)
                        if project.videoCount > 0 {
                            StatusPill(label: "\(project.videoCount) video", tint: MLXRColor.brandPrimary)
                        }
                    }

                    Text(project.subtitle)
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textSecondary)

                    HStack(spacing: MLXRSpacing.md) {
                        projectMetric(label: "Images", value: project.imageCount)
                        projectMetric(label: "Videos", value: project.videoCount)
                        projectMetric(label: "Audio", value: project.audioCount)
                    }
                }

                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, MLXRSpacing.xl)
    }

    private func projectMetric(label: String, value: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            Text("\(value)")
                .font(MLXRType.bodyLarge)
                .fontWeight(.semibold)
                .foregroundStyle(MLXRColor.textPrimary)
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
                    viewerAssetId = nil
                },
                onToggleFavorite: onToggleFavorite,
                onToggleCollection: onToggleCollection,
                onShowAsset: { nextAsset in
                    viewerAssetId = nextAsset.id
                    selectPrimaryAsset(for: nextAsset.id)
                },
                onShowPrevious: previousViewerAction,
                onShowNext: nextViewerAction,
                onClose: {
                    viewerAssetId = nil
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
            viewerAssetId = previousId
            selectedPrimaryAssetId = previousId
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
            viewerAssetId = nextId
            selectedPrimaryAssetId = nextId
        }
    }

    private func openSelectedAsset() {
        guard selectedWorkspaceId != nil else { return }
        guard let selectedPrimaryAssetId else { return }
        viewerAssetId = selectedPrimaryAssetId
    }

    private func selectPrimaryAsset(for assetId: String) {
        selectedPrimaryAssetId = presentation.group(containing: assetId)?.primaryAsset.id ?? assetId
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        guard selectedWorkspaceId != nil else { return }
        let orderedPrimaryAssetIds = presentation.orderedPrimaryAssetIds
        guard !orderedPrimaryAssetIds.isEmpty else { return }
        guard let selectedPrimaryAssetId,
              let currentIndex = orderedPrimaryAssetIds.firstIndex(of: selectedPrimaryAssetId)
        else {
            self.selectedPrimaryAssetId = orderedPrimaryAssetIds.first
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
        self.selectedPrimaryAssetId = orderedPrimaryAssetIds[nextIndex]
    }

    private func removeSelectedImportedAsset() async {
        guard let targetAsset = presentation.viewerAsset(for: viewerAssetId ?? selectedPrimaryAssetId) else {
            return
        }
        guard targetAsset.isImported else {
            return
        }
        await onRemoveImportedAsset(targetAsset.id)
        if viewerAssetId == targetAsset.id {
            viewerAssetId = nil
        }
        if selectedPrimaryAssetId == targetAsset.id {
            selectedPrimaryAssetId = nil
        }
    }

    private func createProjectFromBrowser() {
        let workspace = onCreateWorkspace()
        selectedWorkspaceId = workspace.id
        selectedPrimaryAssetId = nil
        viewerAssetId = nil
    }

    private func applyFocusedRoute() {
        if let focusedWorkspaceId {
            selectedWorkspaceId = focusedWorkspaceId
        }
        if let focusedAssetId {
            selectedPrimaryAssetId = focusedAssetId
            viewerAssetId = focusedAssetId
        }
    }
}
