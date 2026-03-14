import AppKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct LibraryGridView: View {
    let groups: [LibraryAssetGroupPresentation]
    let selectedPrimaryAssetId: String?
    let onSelect: (String) -> Void
    let onOpen: (String) -> Void
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onGridMetricsChange: (Int) -> Void

    private let minTileWidth: CGFloat = 220
    private let maxTileWidth: CGFloat = 300
    private let tileSpacing = MLXRSpacing.md

    var body: some View {
        GeometryReader { proxy in
            let metrics = gridMetrics(for: proxy.size.width)
            ScrollView {
                LazyVGrid(columns: metrics.columns, spacing: tileSpacing) {
                    ForEach(groups) { group in
                        LibraryTileView(
                            group: group,
                            isSelected: selectedPrimaryAssetId == group.primaryAsset.id,
                            width: metrics.tileWidth,
                            onMaterialize: onMaterialize,
                            onSelect: {
                                onSelect(group.primaryAsset.id)
                            },
                            onOpen: {
                                onOpen(group.primaryAsset.id)
                            }
                        )
                    }
                }
                .padding(.horizontal, MLXRSpacing.lg)
                .padding(.bottom, MLXRSpacing.xl)
            }
            .onAppear {
                onGridMetricsChange(metrics.columnCount)
            }
            .onChange(of: proxy.size.width) { _, _ in
                onGridMetricsChange(metrics.columnCount)
            }
        }
    }

    private func gridMetrics(for availableWidth: CGFloat) -> GridMetrics {
        let usableWidth = max(availableWidth - (MLXRSpacing.lg * 2), minTileWidth)
        let columnCount = max(
            Int((usableWidth + tileSpacing) / (minTileWidth + tileSpacing)),
            1
        )
        let tileWidth = min(
            floor(max((usableWidth - CGFloat(columnCount - 1) * tileSpacing) / CGFloat(columnCount), minTileWidth)),
            maxTileWidth
        )
        let columns = Array(
            repeating: GridItem(.fixed(tileWidth), spacing: tileSpacing, alignment: .top),
            count: columnCount
        )
        return GridMetrics(columns: columns, tileWidth: tileWidth, columnCount: columnCount)
    }
}

private struct GridMetrics {
    let columns: [GridItem]
    let tileWidth: CGFloat
    let columnCount: Int
}

private struct LibraryTileView: View {
    let group: LibraryAssetGroupPresentation
    let isSelected: Bool
    let width: CGFloat
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onSelect: () -> Void
    let onOpen: () -> Void

    private let thumbnailAspectRatio: CGFloat = 4.0 / 5.0

    var body: some View {
        GlassCardInteractive {
            VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                LibraryThumbnailView(
                    asset: group.primaryAsset,
                    onMaterialize: onMaterialize
                )
                .frame(maxWidth: .infinity)
                .aspectRatio(thumbnailAspectRatio, contentMode: .fill)
                .overlay(alignment: .topLeading) {
                    HStack(spacing: MLXRSpacing.xs) {
                        StatusPill(
                            label: group.primaryAsset.isImported ? "Imported" : group.primaryAsset.task?.title ?? "Generated",
                            tint: group.primaryAsset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary
                        )
                        if group.assetCount > 1 {
                            StatusPill(label: "\(group.assetCount) in set", tint: MLXRColor.brandSecondary)
                        }
                        if let runState = group.runState {
                            StatusPill(label: label(for: runState), tint: tint(for: runState))
                        }
                    }
                    .padding(MLXRSpacing.sm)
                }
                .overlay(alignment: .topTrailing) {
                    if group.primaryAsset.isFavorite {
                        Image(systemName: "star.fill")
                            .foregroundStyle(MLXRColor.brandWarm)
                            .padding(MLXRSpacing.sm)
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text(group.title)
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(2)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Text(group.summary)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .lineLimit(2)

                    if let filename = group.primaryAsset.filename {
                        Text(filename)
                            .font(MLXRType.captionLarge)
                            .foregroundStyle(MLXRColor.textTertiary)
                            .lineLimit(1)
                    }
                }
                .frame(height: 92, alignment: .topLeading)
            }
            .frame(width: width, alignment: .leading)
            .overlay(alignment: .topTrailing) {
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(MLXRColor.brandPrimary)
                        .padding(MLXRSpacing.sm)
                }
            }
        }
        .frame(width: width, alignment: .leading)
        .contentShape(RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous))
        .onTapGesture(count: 2) {
            onSelect()
            onOpen()
        }
        .onTapGesture {
            onSelect()
        }
        .contextMenu {
            Button("Open") { onOpen() }
            Button("Select") { onSelect() }
        }
    }

    private func tint(for state: RunGroupState) -> Color {
        switch state {
        case .queued:
            MLXRColor.brandWarm
        case .running:
            MLXRColor.brandPrimary
        case .completed:
            MLXRColor.brandSecondary
        case .failed:
            MLXRColor.brandDanger
        case .cancelled:
            MLXRColor.textTertiary
        }
    }

    private func label(for state: RunGroupState) -> String {
        switch state {
        case .queued: "Queued"
        case .running: "Running"
        case .completed: "Completed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
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
                VStack(spacing: MLXRSpacing.xs) {
                    Image(systemName: icon)
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(MLXRColor.textSecondary)
                    Text(asset.kind.rawValue.capitalized)
                        .font(MLXRType.captionSmall)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
        .task(id: asset.id) {
            guard previewImage == nil else { return }
            previewImage = await LibraryThumbnailStore.shared.thumbnail(
                for: asset,
                maxPixelSize: 640,
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
