import AppKit
import AVFoundation
import AVKit
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
    @State private var selectedAssetId: String?
    @State private var previewURL: URL?
    @State private var isLoadingPreview = false
    @State private var isPickingImports = false
    @State private var newCollectionName = ""
    @FocusState private var isLibraryFocused: Bool

    private let tileWidth: CGFloat = 248

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
        HStack(spacing: 0) {
            filterRail
                .frame(minWidth: 250, idealWidth: 280, maxWidth: 320)

            Divider()

            VStack(spacing: 0) {
                libraryToolbar
                centerPane
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if selectedAsset != nil {
                Divider()
                detailPane
                    .frame(minWidth: 360, idealWidth: 400, maxWidth: 430)
            }
        }
        .background(AdaptiveBackground())
        .focusable()
        .focused($isLibraryFocused)
        .onChange(of: selectedAssetId) { _, _ in
            Task { await loadPreviewForSelection() }
        }
        .onChange(of: filteredAssets.map(\.id)) { _, visibleIds in
            if let selectedAssetId, !visibleIds.contains(selectedAssetId) {
                self.selectedAssetId = nil
            }
        }
        .onAppear {
            isLibraryFocused = true
        }
        .onExitCommand {
            selectedAssetId = nil
        }
        .onMoveCommand { direction in
            moveSelection(direction)
        }
        .fileImporter(
            isPresented: $isPickingImports,
            allowedContentTypes: [.image, .movie, .audio],
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result else { return }
            Task {
                let imported = await onImportAssets(urls)
                selectedAssetId = imported.first?.id
            }
        }
    }

    private var libraryToolbar: some View {
        HStack(spacing: MLXRSpacing.sm) {
            TextField("Search prompts, models, or filenames", text: $query)
                .textFieldStyle(.roundedBorder)

            Button("Import from Finder") {
                isPickingImports = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, MLXRSpacing.lg)
        .padding(.vertical, MLXRSpacing.md)
    }

    private var filterRail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                FeatureHeader(
                    eyebrow: "Library",
                    title: "Everything you made or imported",
                    subtitle: "The library is your studio bin: browse assets, group them by collection, and send them straight back into Studio."
                )

                GlassCard(title: "Filter", subtitle: "Narrow the library to the media you need right now.") {
                    filterSection("Media") {
                        ForEach(LibraryAssetFilter.allCases) { filter in
                            filterButton(
                                title: filter.rawValue,
                                isSelected: selectedFilter == filter,
                                count: assets.filter { filter.includes($0) }.count
                            ) {
                                selectedFilter = filter
                            }
                        }
                    }

                    Toggle("Favorites only", isOn: $favoritesOnly)
                        .toggleStyle(.switch)

                    Picker("Model", selection: $selectedModelId) {
                        Text("All Models").tag("all")
                        ForEach(modelOptions, id: \.self) { modelId in
                            Text(modelId).tag(modelId)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Workflow", selection: $selectedTaskRaw) {
                        Text("All Workflows").tag("all")
                        ForEach(taskOptions, id: \.rawValue) { task in
                            Text(task.title).tag(task.rawValue)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Sort", selection: $selectedSort) {
                        ForEach(AssetSortMode.allCases, id: \.rawValue) { sort in
                            Text(sort.rawValue).tag(sort)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                GlassCard(title: "Collections", subtitle: "Keep lightweight working sets without losing the global library.") {
                    filterButton(
                        title: "All Assets",
                        isSelected: selectedCollectionId == nil,
                        count: assets.count
                    ) {
                        selectedCollectionId = nil
                    }

                    ForEach(collections) { collection in
                        filterButton(
                            title: collection.title,
                            isSelected: selectedCollectionId == collection.id,
                            count: assets.filter { $0.collectionIds.contains(collection.id) }.count
                        ) {
                            selectedCollectionId = collection.id
                        }
                    }

                    HStack(spacing: MLXRSpacing.xs) {
                        TextField("New collection", text: $newCollectionName)
                            .textFieldStyle(.roundedBorder)
                        Button("Add") {
                            let trimmed = newCollectionName.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            onCreateCollection(trimmed)
                            newCollectionName = ""
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding(.horizontal, MLXRSpacing.lg)
            .padding(.vertical, MLXRSpacing.xl)
        }
    }

    private var centerPane: some View {
        Group {
            if groupedAssets.isEmpty {
                EmptyStateView(
                    title: "Nothing matches yet",
                    subtitle: "Import something or make something in Studio, and it will show up here as a reusable asset.",
                    systemImage: "photo.stack"
                )
                .padding(MLXRSpacing.xl)
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: tileWidth, maximum: tileWidth), spacing: MLXRSpacing.md)], spacing: MLXRSpacing.md) {
                        ForEach(groupedAssets) { group in
                            assetGroupCard(group)
                        }
                    }
                    .padding(.horizontal, MLXRSpacing.lg)
                    .padding(.bottom, MLXRSpacing.xl)
                }
            }
        }
    }

    private func assetGroupCard(_ group: LibraryAssetGroup) -> some View {
        let asset = group.primaryAsset
        let isSelected = selectedAssetId == asset.id
        return Button {
            selectedAssetId = asset.id
        } label: {
            GlassCardInteractive {
                VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                    LibraryThumbnailView(asset: asset, onMaterialize: onMaterialize)
                        .frame(height: 164)

                    HStack {
                        StatusPill(
                            label: asset.isImported ? "Imported" : asset.task?.title ?? "Generated",
                            tint: asset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary
                        )
                        if group.assets.count > 1 {
                            StatusPill(label: "\(group.assets.count) in set", tint: MLXRColor.brandSecondary)
                        }
                        Spacer()
                        if asset.isFavorite {
                            Image(systemName: "star.fill")
                                .foregroundStyle(MLXRColor.brandWarm)
                        }
                    }

                    Text(asset.title)
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(1)
                    if let runGroupTitle = runGroupTitle(for: group) {
                        Text(runGroupTitle)
                            .font(MLXRType.captionLarge)
                            .foregroundStyle(MLXRColor.textTertiary)
                            .lineLimit(1)
                    }
                    Text(asset.subtitle)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .lineLimit(2)
                    Text(asset.sourceSummary)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .overlay(alignment: .topTrailing) {
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(MLXRColor.brandPrimary)
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var detailPane: some View {
        if let selectedAsset {
            ScrollView {
                VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                    GlassCard(
                        title: selectedAsset.title,
                        subtitle: selectedAsset.subtitle
                    ) {
                        HStack {
                            Spacer()
                            Button {
                                selectedAssetId = nil
                            } label: {
                                Label("Close", systemImage: "xmark.circle.fill")
                            }
                            .buttonStyle(.bordered)
                        }

                        HStack(spacing: MLXRSpacing.xs) {
                            StatusPill(label: selectedAsset.sourceSummary, tint: selectedAsset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary)
                            if let task = selectedAsset.task {
                                StatusPill(label: task.title, tint: MLXRColor.brandSecondary)
                            }
                        }

                        previewView(for: selectedAsset)
                            .frame(maxWidth: .infinity, minHeight: 320)

                        actionButtons(for: selectedAsset)
                        metadataRows(for: selectedAsset)
                        collectionRows(for: selectedAsset)

                        if let group = selectedGroup, group.assets.count > 1 {
                            relatedAssetsStrip(group.assets, title: runGroupTitle(for: group) ?? "This set")
                        }
                    }
                }
                .padding(.horizontal, MLXRSpacing.lg)
                .padding(.vertical, MLXRSpacing.xl)
            }
        }
    }

    @ViewBuilder
    private func actionButtons(for asset: LibraryAsset) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            HStack(spacing: MLXRSpacing.sm) {
                Button(asset.isFavorite ? "Unfavorite" : "Favorite") {
                    onToggleFavorite(asset.id)
                }
                .buttonStyle(.bordered)

                if let previewURL {
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([previewURL])
                    }
                    .buttonStyle(.bordered)

                    Button("Open") {
                        NSWorkspace.shared.open(previewURL)
                    }
                    .buttonStyle(.bordered)
                }
            }

            HStack(spacing: MLXRSpacing.sm) {
                if asset.isImage {
                    Button("Edit") {
                        onOpenInStudio(StudioOpenRequest(task: .imageEdit, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Animate") {
                        onOpenInStudio(StudioOpenRequest(task: .videoConditionImage, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                    }
                    .buttonStyle(.bordered)
                } else if asset.isVideo {
                    Button("Guide") {
                        onOpenInStudio(StudioOpenRequest(task: .videoConditionVideo, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Retake") {
                        onOpenInStudio(StudioOpenRequest(task: .videoRetake, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                    }
                    .buttonStyle(.bordered)
                } else if asset.isAudio {
                    Button("Use in Studio") {
                        onOpenInStudio(StudioOpenRequest(task: .videoConditionAudio, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                    }
                    .buttonStyle(.borderedProminent)
                }

                Button("Use as reference") {
                    let task: ProductTask = asset.isVideo ? .videoConditionVideo : asset.isAudio ? .videoConditionAudio : .imageEdit
                    onOpenInStudio(StudioOpenRequest(task: task, prompt: asset.prompt, focusedAssetId: asset.id, referenceAssetIds: [asset.id]))
                }
                .buttonStyle(.bordered)
            }

            if asset.isImported {
                Button("Remove from library") {
                    Task { await onRemoveImportedAsset(asset.id) }
                }
                .buttonStyle(.bordered)
            }
        }
    }

    @ViewBuilder
    private func metadataRows(for asset: LibraryAsset) -> some View {
        DetailRow(label: "Type", value: asset.kind.rawValue.capitalized)
        DetailRow(label: "Source", value: asset.sourceSummary)
        if let modelId = asset.modelId {
            DetailRow(label: "Model", value: modelId)
        }
        if let task = asset.task {
            DetailRow(label: "Workflow", value: task.title)
        }
        if let importedAsset = asset.importedAsset {
            DetailRow(label: "Original path", value: importedAsset.sourcePath)
        }
    }

    @ViewBuilder
    private func collectionRows(for asset: LibraryAsset) -> some View {
        if !collections.isEmpty {
            VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                Text("Collections")
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(MLXRColor.textTertiary)
                ForEach(collections) { collection in
                    Toggle(
                        isOn: Binding(
                            get: { asset.collectionIds.contains(collection.id) },
                            set: { _ in onToggleCollection(asset.id, collection.id) }
                        )
                    ) {
                        Text(collection.title)
                            .font(MLXRType.bodySmall)
                    }
                    .toggleStyle(.checkbox)
                }
            }
        }
    }

    private func relatedAssetsStrip(_ assets: [LibraryAsset], title: String) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(title)
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: MLXRSpacing.sm) {
                    ForEach(assets) { asset in
                        Button {
                            selectedAssetId = asset.id
                        } label: {
                            LibraryThumbnailView(asset: asset, onMaterialize: onMaterialize)
                                .frame(width: 120, height: 84)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func previewView(for asset: LibraryAsset) -> some View {
        if isLoadingPreview {
            IndeterminateProgress(label: "Loading preview…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let previewURL {
            if asset.isImage, let image = NSImage(contentsOf: previewURL) {
                MediaHero(image: image, dominantHue: dominantHue(from: image))
            } else if asset.isVideo {
                LibraryVideoPreview(url: previewURL)
            } else {
                EmptyStateView(
                    title: asset.title,
                    subtitle: asset.subtitle,
                    systemImage: icon(for: asset)
                )
            }
        } else {
            EmptyStateView(
                title: "Preview not ready",
                subtitle: "Select an asset to load a local preview.",
                systemImage: icon(for: asset)
            )
        }
    }

    private var filteredAssets: [LibraryAsset] {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return assets.filter { asset in
            guard selectedFilter.includes(asset) else { return false }
            guard !favoritesOnly || asset.isFavorite else { return false }
            guard selectedModelId == "all" || asset.modelId == selectedModelId else { return false }
            guard selectedTaskRaw == "all" || asset.task?.rawValue == selectedTaskRaw else { return false }
            if let selectedCollectionId, !asset.collectionIds.contains(selectedCollectionId) {
                return false
            }
            guard trimmedQuery.isEmpty || asset.searchableText.contains(trimmedQuery) else { return false }
            return true
        }
        .sorted(by: sortComparator)
    }

    private var groupedAssets: [LibraryAssetGroup] {
        let grouped = Dictionary(grouping: filteredAssets) { asset in
            asset.runGroupId ?? asset.id
        }
        return grouped.values.map { assets in
            let sortedAssets = assets.sorted(by: sortComparator)
            return LibraryAssetGroup(id: sortedAssets.first?.runGroupId ?? sortedAssets.first?.id ?? UUID().uuidString, assets: sortedAssets)
        }
        .sorted { lhs, rhs in
            sortComparator(lhs.primaryAsset, rhs.primaryAsset)
        }
    }

    private var selectedAsset: LibraryAsset? {
        guard let selectedAssetId else { return nil }
        return filteredAssets.first(where: { $0.id == selectedAssetId }) ?? assets.first(where: { $0.id == selectedAssetId })
    }

    private var selectedGroup: LibraryAssetGroup? {
        guard let selectedAsset else { return nil }
        return groupedAssets.first { group in group.assets.contains(selectedAsset) }
    }

    private var orderedPrimaryAssetIds: [String] {
        groupedAssets.map { $0.primaryAsset.id }
    }

    private var modelOptions: [String] {
        Array(Set(assets.compactMap(\.modelId))).sorted()
    }

    private var taskOptions: [ProductTask] {
        Array(Set(assets.compactMap(\.task))).sorted { $0.title < $1.title }
    }

    private func sortComparator(_ lhs: LibraryAsset, _ rhs: LibraryAsset) -> Bool {
        switch selectedSort {
        case .newest:
            let lhsDate = lhs.createdAt
            let rhsDate = rhs.createdAt
            if lhsDate != rhsDate { return lhsDate > rhsDate }
            return lhs.title < rhs.title
        case .lastUsed:
            let lhsDate = lhs.lastUsedAt ?? lhs.createdAt
            let rhsDate = rhs.lastUsedAt ?? rhs.createdAt
            if lhsDate != rhsDate { return lhsDate > rhsDate }
            return lhs.title < rhs.title
        }
    }

    private func filterSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(title.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            content()
        }
    }

    private func filterButton(title: String, isSelected: Bool, count: Int, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(title)
                Spacer()
                Text("\(count)")
                    .foregroundStyle(MLXRColor.textTertiary)
            }
            .font(MLXRType.bodySmall)
            .foregroundStyle(isSelected ? MLXRColor.brandPrimary : MLXRColor.textSecondary)
            .padding(.vertical, MLXRSpacing.xxs)
        }
        .buttonStyle(.plain)
    }

    private func loadPreviewForSelection() async {
        guard let selectedAsset else {
            previewURL = nil
            isLoadingPreview = false
            return
        }
        isLoadingPreview = true
        previewURL = await onMaterialize(selectedAsset)
        isLoadingPreview = false
    }

    private func icon(for asset: LibraryAsset) -> String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }

    private func runGroupTitle(for group: LibraryAssetGroup) -> String? {
        guard let runGroupId = group.assets.first?.runGroupId else { return nil }
        return runGroups.first(where: { $0.id == runGroupId })?.title
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        let orderedIds = orderedPrimaryAssetIds
        guard !orderedIds.isEmpty else { return }

        guard let selectedAssetId, let currentIndex = orderedIds.firstIndex(of: selectedAssetId) else {
            self.selectedAssetId = orderedIds.first
            return
        }

        let nextIndex: Int
        switch direction {
        case .left, .up:
            nextIndex = max(currentIndex - 1, 0)
        case .right, .down:
            nextIndex = min(currentIndex + 1, orderedIds.count - 1)
        @unknown default:
            nextIndex = currentIndex
        }
        self.selectedAssetId = orderedIds[nextIndex]
    }
}

private struct LibraryAssetGroup: Identifiable {
    let id: String
    let assets: [LibraryAsset]

    var primaryAsset: LibraryAsset {
        assets.first!
    }
}

private struct LibraryThumbnailView: View {
    let asset: LibraryAsset
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?

    @State private var previewImage: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(MLXRColor.surfaceCard)

            if let previewImage {
                Image(nsImage: previewImage)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: icon)
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(MLXRColor.textSecondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
        .frame(maxWidth: .infinity)
        .task(id: asset.id) {
            guard previewImage == nil else { return }
            guard let previewURL = await onMaterialize(asset) else { return }
            if asset.isImage {
                previewImage = NSImage(contentsOf: previewURL)
            } else if asset.isVideo {
                previewImage = await videoThumbnail(for: previewURL)
            }
        }
    }

    private var icon: String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }

    private func videoThumbnail(for url: URL) async -> NSImage? {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 640, height: 640)
        let time = NSValue(time: .zero)
        return await withCheckedContinuation { continuation in
            generator.generateCGImagesAsynchronously(forTimes: [time]) { _, cgImage, _, _, _ in
                if let cgImage {
                    continuation.resume(returning: NSImage(cgImage: cgImage, size: .zero))
                } else {
                    continuation.resume(returning: nil)
                }
            }
        }
    }
}

private struct LibraryVideoPreview: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        view.videoGravity = .resizeAspect
        view.player = AVPlayer(url: url)
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        let currentURL = (nsView.player?.currentItem?.asset as? AVURLAsset)?.url
        if currentURL != url {
            nsView.player = AVPlayer(url: url)
        }
    }

    static func dismantleNSView(_ nsView: AVPlayerView, coordinator: ()) {
        nsView.player?.pause()
        nsView.player = nil
    }
}
