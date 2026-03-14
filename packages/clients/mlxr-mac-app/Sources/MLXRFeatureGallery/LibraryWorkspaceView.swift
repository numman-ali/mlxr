import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI
import UniformTypeIdentifiers

public struct LibraryWorkspaceView: View {
    private let assets: [LibraryAsset]
    private let runGroups: [RunGroupRecord]
    private let collections: [CollectionRecord]
    private let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    private let onImportAssets: @Sendable ([URL]) async -> [ImportedAssetRecord]
    private let onRemoveImportedAsset: @Sendable (String) async -> Void
    private let onOpenInStudio: (StudioOpenRequest) -> Void
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
    @State private var selectedPrimaryAssetId: String?
    @State private var viewerAssetId: String?
    @State private var isPickingImports = false
    @State private var newCollectionName = ""
    @State private var gridColumnCount = 1
    @FocusState private var isLibraryFocused: Bool

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
        ZStack {
            HStack(spacing: 0) {
                LibraryFilterRail(
                    selectedFilter: $selectedFilter,
                    selectedModelId: $selectedModelId,
                    selectedTaskRaw: $selectedTaskRaw,
                    selectedSort: $selectedSort,
                    favoritesOnly: $favoritesOnly,
                    selectedCollectionId: $selectedCollectionId,
                    newCollectionName: $newCollectionName,
                    presentation: presentation,
                    onCreateCollection: onCreateCollection
                )
                .frame(minWidth: 250, idealWidth: 280, maxWidth: 320)

                Divider()

                VStack(spacing: 0) {
                    libraryToolbar
                    content
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

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
        }
        .onChange(of: presentation.orderedPrimaryAssetIds) { _, visiblePrimaryIds in
            if let selectedPrimaryAssetId, !visiblePrimaryIds.contains(selectedPrimaryAssetId) {
                self.selectedPrimaryAssetId = nil
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
            } else {
                selectedPrimaryAssetId = nil
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
            runGroups: runGroups,
            collections: collections,
            filters: currentFilters
        )
    }

    private var viewerAsset: LibraryAsset? {
        presentation.viewerAsset(for: viewerAssetId)
    }

    private var libraryToolbar: some View {
        HStack(spacing: MLXRSpacing.sm) {
            TextField("Search prompts, models, or filenames", text: $query)
                .textFieldStyle(.roundedBorder)

            if viewerAsset != nil {
                Button("Close Preview") {
                    viewerAssetId = nil
                }
                .buttonStyle(.bordered)
            } else if selectedPrimaryAssetId != nil {
                Button("Open") {
                    openSelectedAsset()
                }
                .buttonStyle(.bordered)

                Button("Clear Selection") {
                    selectedPrimaryAssetId = nil
                }
                .buttonStyle(.bordered)
            }

            Spacer(minLength: 0)

            Text("\(presentation.filteredAssets.count) asset\(presentation.filteredAssets.count == 1 ? "" : "s")")
                .font(MLXRType.captionLarge)
                .foregroundStyle(MLXRColor.textTertiary)

            Button("Import from Finder") {
                isPickingImports = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, MLXRSpacing.lg)
        .padding(.vertical, MLXRSpacing.md)
    }

    @ViewBuilder
    private var content: some View {
        if presentation.groups.isEmpty {
            EmptyStateView(
                title: "Nothing matches yet",
                subtitle: "Import something or make something in Studio, and it will show up here as a reusable asset.",
                systemImage: "photo.stack"
            )
            .padding(MLXRSpacing.xl)
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
                onOpenInStudio: onOpenInStudio,
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
        guard let selectedPrimaryAssetId else { return }
        viewerAssetId = selectedPrimaryAssetId
    }

    private func selectPrimaryAsset(for assetId: String) {
        selectedPrimaryAssetId = presentation.group(containing: assetId)?.primaryAsset.id ?? assetId
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
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
}
