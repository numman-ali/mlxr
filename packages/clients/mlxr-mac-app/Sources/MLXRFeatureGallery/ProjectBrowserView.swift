import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct ProjectBrowserView: View {
    let projects: [ProjectSummaryPresentation]
    let selectedProjectId: String?
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onOpenProject: (String) -> Void
    let onGridMetricsChange: (Int) -> Void
    let onCreateProject: () -> Void
    @Environment(\.mlxrBottomOverlayInset) private var bottomOverlayInset

    private let minCardWidth: CGFloat = 188
    private let maxCardWidth: CGFloat = 236

    var body: some View {
        if projects.isEmpty {
            EmptyStateView(
                title: "No projects yet",
                subtitle: "Create from Home or Library and MLXR will keep each creative thread together here.",
                systemImage: "rectangle.stack.badge.plus"
            )
            .padding(.horizontal, MLXRSpacing.xl)
            .padding(.top, MLXRSpacing.xl)
            .padding(.bottom, max(MLXRSpacing.xl, bottomOverlayInset))
        } else {
            GeometryReader { proxy in
                let metrics = gridMetrics(for: proxy.size.width)
                ScrollView {
                    LazyVGrid(columns: metrics.columns, spacing: MLXRSpacing.md) {
                        ForEach(projects) { project in
                            ProjectCardView(
                                project: project,
                                isSelected: selectedProjectId == project.id || project.isActive,
                                width: metrics.cardWidth,
                                onMaterialize: onMaterialize
                            ) {
                                onOpenProject(project.id)
                            }
                        }
                    }
                    .padding(.horizontal, MLXRSpacing.xl)
                    .padding(.top, MLXRSpacing.md)
                    .padding(.bottom, max(MLXRSpacing.xl, bottomOverlayInset))
                }
                .onAppear {
                    onGridMetricsChange(metrics.columns.count)
                }
                .onChange(of: metrics.columns.count) { _, newValue in
                    onGridMetricsChange(newValue)
                }
            }
        }
    }

    private func gridMetrics(for availableWidth: CGFloat) -> (columns: [GridItem], cardWidth: CGFloat) {
        let usableWidth = max(availableWidth - (MLXRSpacing.xl * 2), minCardWidth)
        let columnCount = max(Int((usableWidth + MLXRSpacing.md) / (minCardWidth + MLXRSpacing.md)), 1)
        let cardWidth = min(
            floor(max((usableWidth - CGFloat(columnCount - 1) * MLXRSpacing.md) / CGFloat(columnCount), minCardWidth)),
            maxCardWidth
        )
        let columns = Array(
            repeating: GridItem(.fixed(cardWidth), spacing: MLXRSpacing.md, alignment: .top),
            count: columnCount
        )
        return (columns, cardWidth)
    }
}

private struct ProjectCardView: View {
    let project: ProjectSummaryPresentation
    let isSelected: Bool
    let width: CGFloat
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onOpen: () -> Void

    private let heroAspectRatio: CGFloat = 1

    var body: some View {
        MediaTileSurface(isSelected: isSelected, cornerRadius: MLXRRadius.lg, action: onOpen) {
            VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                ZStack(alignment: .topTrailing) {
                    if let heroAsset = project.heroAsset {
                        LibraryAssetThumbnailView(asset: heroAsset, onMaterialize: onMaterialize)
                            .frame(width: width, height: width)
                            .aspectRatio(heroAspectRatio, contentMode: .fill)
                    } else {
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .fill(MLXRColor.surfaceCard)
                            .frame(width: width, height: width)
                            .aspectRatio(heroAspectRatio, contentMode: .fill)
                            .overlay {
                                VStack(spacing: MLXRSpacing.xs) {
                                    Image(systemName: "sparkles.rectangle.stack")
                                        .font(.system(size: 24, weight: .semibold))
                                        .foregroundStyle(MLXRColor.brandPrimary)
                                    Text("Ready for a first result")
                                        .font(MLXRType.captionLarge)
                                        .foregroundStyle(MLXRColor.textSecondary)
                                }
                            }
                    }

                    if project.isActive {
                        StatusPill(label: "Active", tint: MLXRColor.brandPrimary)
                            .padding(MLXRSpacing.sm)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous))

                VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                    Text(project.title)
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(1)
                    Text(project.subtitle)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .lineLimit(2)
                }

                HStack(spacing: MLXRSpacing.xs) {
                    StatusPill(label: "\(project.assetCount)", tint: MLXRColor.brandSecondary)
                    if project.runGroupCount > 0 {
                        StatusPill(label: "\(project.runGroupCount) sets", tint: MLXRColor.brandWarm)
                    }
                }
            }
            .frame(width: width, alignment: .leading)
        }
        .frame(width: width, alignment: .leading)
    }
}
