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

    private let minTileWidth: CGFloat = 164
    private let maxTileWidth: CGFloat = 228
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

    private let thumbnailAspectRatio: CGFloat = 1

    var body: some View {
        MediaTileSurface(
            isSelected: isSelected,
            cornerRadius: MLXRRadius.md,
            action: onOpen
        ) {
            ZStack(alignment: .topTrailing) {
                LibraryAssetThumbnailView(
                    asset: group.primaryAsset,
                    onMaterialize: onMaterialize
                )
                .frame(width: width, height: width)
                .aspectRatio(thumbnailAspectRatio, contentMode: .fill)
                .clipped()

                VStack(alignment: .trailing, spacing: MLXRSpacing.xs) {
                    if group.primaryAsset.isFavorite {
                        Image(systemName: "star.fill")
                            .foregroundStyle(MLXRColor.brandWarm)
                            .padding(.top, MLXRSpacing.sm)
                            .padding(.trailing, MLXRSpacing.sm)
                    }

                    Spacer()

                    if group.assetCount > 1 {
                        Text("\(group.assetCount)")
                            .font(MLXRType.captionLarge)
                            .fontWeight(.semibold)
                            .foregroundStyle(MLXRColor.textPrimary)
                            .padding(.horizontal, MLXRSpacing.sm)
                            .padding(.vertical, MLXRSpacing.xs)
                            .background(.ultraThinMaterial, in: Capsule())
                            .padding(.trailing, MLXRSpacing.sm)
                            .padding(.bottom, MLXRSpacing.sm)
                    }
                }
            }
        }
        .frame(width: width, alignment: .leading)
        .contextMenu {
            Button("Open") { onOpen() }
            Button("Select") { onSelect() }
        }
        .accessibilityLabel(accessibilityLabel)
    }

    private var accessibilityLabel: String {
        if group.assetCount > 1 {
            "\(group.title), \(group.assetCount) items"
        } else {
            group.title
        }
    }
}
