import AppKit
import AVKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct LibraryViewerSheet: View {
    let asset: LibraryAsset
    let group: LibraryAssetGroupPresentation?
    let collections: [CollectionRecord]
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onRemoveImportedAsset: @Sendable (String) async -> Void
    let onSeedComposer: (ComposerSeedRequest) -> Void
    let onSetProjectCover: (() -> Void)?
    let onToggleFavorite: (String) -> Void
    let onToggleCollection: (String, String) -> Void
    let onShowAsset: (LibraryAsset) -> Void
    let onShowPrevious: (() -> Void)?
    let onShowNext: (() -> Void)?
    let onClose: () -> Void

    @State private var previewURL: URL?
    @State private var isLoadingPreview = false

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                header
                previewSection
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                    metadataHeader
                    actionButtons
                    metadataRows
                    collectionsSection
                    relatedAssetsSection
                }
                .padding(.horizontal, MLXRSpacing.xl)
                .padding(.vertical, MLXRSpacing.xl)
            }
            .frame(width: 360)
            .background(MLXRColor.canvasRaised)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(AdaptiveBackground())
        .task(id: asset.id) {
            await loadPreview()
        }
    }

    private var header: some View {
        HStack(spacing: MLXRSpacing.sm) {
            Button {
                onShowPrevious?()
            } label: {
                Label("Previous", systemImage: "chevron.left")
            }
            .buttonStyle(.bordered)
            .disabled(onShowPrevious == nil)

            Button {
                onShowNext?()
            } label: {
                Label("Next", systemImage: "chevron.right")
            }
            .buttonStyle(.bordered)
            .disabled(onShowNext == nil)

            Spacer()

            Button("Close") {
                onClose()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.top, MLXRSpacing.lg)
        .padding(.bottom, MLXRSpacing.md)
    }

    @ViewBuilder
    private var previewSection: some View {
        ZStack {
            if isLoadingPreview {
                IndeterminateProgress(label: "Loading preview…")
            } else if let previewURL {
                if asset.isImage, let image = NSImage(contentsOf: previewURL) {
                    MediaHero(image: image, dominantHue: dominantHue(from: image))
                } else if asset.isVideo {
                    LibraryVideoPreview(url: previewURL)
                } else {
                EmptyStateView(
                    title: asset.displayTitle,
                    subtitle: asset.subtitle,
                    systemImage: icon(for: asset)
                )
                }
            } else {
                EmptyStateView(
                    title: "Preview not ready",
                    subtitle: "MLXR could not materialize a local preview for this asset yet.",
                    systemImage: icon(for: asset)
                )
            }
        }
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.bottom, MLXRSpacing.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var metadataHeader: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            HStack(spacing: MLXRSpacing.xs) {
                StatusPill(
                    label: asset.sourceSummary,
                    tint: asset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary
                )
                if let task = asset.task {
                    StatusPill(label: task.title, tint: MLXRColor.brandSecondary)
                }
                if let group, group.assetCount > 1 {
                    StatusPill(label: "\(group.assetCount) in set", tint: MLXRColor.brandSecondary)
                }
            }

            Text(asset.displayTitle)
                .font(MLXRType.titleLarge)
                .foregroundStyle(MLXRColor.textPrimary)

            Text(asset.subtitle)
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textSecondary)
        }
    }

    private var actionButtons: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            HStack(spacing: MLXRSpacing.sm) {
                Button(asset.isFavorite ? "Unfavorite" : "Favorite") {
                    onToggleFavorite(asset.id)
                }
                .buttonStyle(.bordered)

                if let onSetProjectCover {
                    Button("Set as project cover", action: onSetProjectCover)
                        .buttonStyle(.bordered)
                }

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
                        onSeedComposer(
                            ComposerSeedRequest(
                                workspaceId: asset.workspaceId,
                                task: .imageEdit,
                                prompt: asset.prompt,
                                focusedAssetId: asset.id,
                                referenceAssetIds: [asset.id]
                            )
                        )
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Animate") {
                        onSeedComposer(
                            ComposerSeedRequest(
                                workspaceId: asset.workspaceId,
                                task: .videoConditionImage,
                                prompt: asset.prompt,
                                focusedAssetId: asset.id,
                                referenceAssetIds: [asset.id]
                            )
                        )
                    }
                    .buttonStyle(.bordered)
                } else if asset.isVideo {
                    Button("Guide") {
                        onSeedComposer(
                            ComposerSeedRequest(
                                workspaceId: asset.workspaceId,
                                task: .videoConditionVideo,
                                prompt: asset.prompt,
                                focusedAssetId: asset.id,
                                referenceAssetIds: [asset.id]
                            )
                        )
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Retake") {
                        onSeedComposer(
                            ComposerSeedRequest(
                                workspaceId: asset.workspaceId,
                                task: .videoRetake,
                                prompt: asset.prompt,
                                focusedAssetId: asset.id,
                                referenceAssetIds: [asset.id]
                            )
                        )
                    }
                    .buttonStyle(.bordered)
                } else if asset.isAudio {
                    Button("Use as audio guide") {
                        onSeedComposer(
                            ComposerSeedRequest(
                                workspaceId: asset.workspaceId,
                                task: .videoConditionAudio,
                                prompt: asset.prompt,
                                focusedAssetId: asset.id,
                                referenceAssetIds: [asset.id]
                            )
                        )
                    }
                    .buttonStyle(.borderedProminent)
                }

                Button("Use as reference") {
                    let task: ProductTask =
                        if asset.isVideo {
                            .videoConditionVideo
                        } else if asset.isAudio {
                            .videoConditionAudio
                        } else {
                            .imageEdit
                        }
                    onSeedComposer(
                        ComposerSeedRequest(
                            workspaceId: asset.workspaceId,
                            task: task,
                            prompt: asset.prompt,
                            focusedAssetId: asset.id,
                            referenceAssetIds: [asset.id]
                        )
                    )
                }
                .buttonStyle(.bordered)
            }

            if asset.isImported {
                Button("Remove from library") {
                    Task {
                        await onRemoveImportedAsset(asset.id)
                        onClose()
                    }
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var metadataRows: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            DetailRow(label: "Type", value: asset.kind.rawValue.capitalized)
            DetailRow(label: "Source", value: asset.sourceSummary)
            if let filename = asset.filename {
                DetailRow(label: "Filename", value: filename)
            }
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
    }

    @ViewBuilder
    private var collectionsSection: some View {
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

    @ViewBuilder
    private var relatedAssetsSection: some View {
        if let group, group.assetCount > 1 {
            VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                Text("This set")
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(MLXRColor.textTertiary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: MLXRSpacing.sm) {
                        ForEach(group.assets) { relatedAsset in
                            Button {
                                onShowAsset(relatedAsset)
                            } label: {
                                LibraryViewerThumbnail(
                                    asset: relatedAsset,
                                    isSelected: relatedAsset.id == asset.id,
                                    onMaterialize: onMaterialize
                                )
                                .frame(width: 132, height: 92)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
    }

    private func icon(for asset: LibraryAsset) -> String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }

    private func loadPreview() async {
        isLoadingPreview = true
        previewURL = await onMaterialize(asset)
        isLoadingPreview = false
    }
}

private struct LibraryViewerThumbnail: View {
    let asset: LibraryAsset
    let isSelected: Bool
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
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(MLXRColor.textSecondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .strokeBorder(isSelected ? MLXRColor.brandPrimary : .clear, lineWidth: 2)
        }
        .task(id: asset.id) {
            previewImage = await LibraryThumbnailStore.shared.thumbnail(
                for: asset,
                maxPixelSize: 320,
                materialize: onMaterialize
            )
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
