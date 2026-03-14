import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct ProjectBrowserView: View {
    let projects: [ProjectSummaryPresentation]
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onOpenProject: (String) -> Void
    let onCreateProject: () -> Void

    private let minCardWidth: CGFloat = 220
    private let maxCardWidth: CGFloat = 280

    var body: some View {
        if projects.isEmpty {
            EmptyStateView(
                title: "No projects yet",
                subtitle: "Create from Home or Library and MLXR will keep each creative thread together here.",
                systemImage: "rectangle.stack.badge.plus"
            )
            .padding(MLXRSpacing.xl)
        } else {
            GeometryReader { proxy in
                let metrics = gridMetrics(for: proxy.size.width)
                ScrollView {
                    VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                        HStack {
                            VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                                Text("Projects")
                                    .font(MLXRType.titleLarge)
                                    .foregroundStyle(MLXRColor.textPrimary)
                                Text("Open a project to keep prompting, browsing, and iterating in one place.")
                                    .font(MLXRType.bodyMedium)
                                    .foregroundStyle(MLXRColor.textSecondary)
                            }
                            Spacer()
                            Button("New Project", action: onCreateProject)
                                .buttonStyle(.borderedProminent)
                        }
                        .padding(.horizontal, MLXRSpacing.xl)
                        .padding(.top, MLXRSpacing.xl)

                        LazyVGrid(columns: metrics.columns, spacing: MLXRSpacing.lg) {
                            ForEach(projects) { project in
                                ProjectCardView(
                                    project: project,
                                    width: metrics.cardWidth,
                                    onMaterialize: onMaterialize
                                ) {
                                    onOpenProject(project.id)
                                }
                            }
                        }
                        .padding(.horizontal, MLXRSpacing.xl)
                        .padding(.bottom, MLXRSpacing.xl)
                    }
                }
            }
        }
    }

    private func gridMetrics(for availableWidth: CGFloat) -> (columns: [GridItem], cardWidth: CGFloat) {
        let usableWidth = max(availableWidth - (MLXRSpacing.xl * 2), minCardWidth)
        let columnCount = max(Int((usableWidth + MLXRSpacing.lg) / (minCardWidth + MLXRSpacing.lg)), 1)
        let cardWidth = min(
            floor(max((usableWidth - CGFloat(columnCount - 1) * MLXRSpacing.lg) / CGFloat(columnCount), minCardWidth)),
            maxCardWidth
        )
        let columns = Array(
            repeating: GridItem(.fixed(cardWidth), spacing: MLXRSpacing.lg, alignment: .top),
            count: columnCount
        )
        return (columns, cardWidth)
    }
}

private struct ProjectCardView: View {
    let project: ProjectSummaryPresentation
    let width: CGFloat
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onOpen: () -> Void

    private let heroAspectRatio: CGFloat = 1

    var body: some View {
        MediaTileSurface(isSelected: project.isActive, cornerRadius: MLXRRadius.lg, action: onOpen) {
            VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
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
                                        .font(.system(size: 28, weight: .semibold))
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

                VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                    Text(project.title)
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(1)
                    Text(project.subtitle)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .lineLimit(2)
                }

                HStack(spacing: MLXRSpacing.sm) {
                    StatusPill(label: "\(project.assetCount) assets", tint: MLXRColor.brandSecondary)
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
