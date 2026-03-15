import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct ProjectHeaderView: View {
    let project: ProjectSummaryPresentation
    let selectedAsset: LibraryAsset?
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?
    let onSetCover: (() -> Void)?

    var body: some View {
        HStack(alignment: .center, spacing: MLXRSpacing.md) {
            projectCover

            VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                Text(project.title)
                    .font(MLXRType.titleLarge)
                    .foregroundStyle(MLXRColor.textPrimary)

                Text(project.subtitle)
                    .font(MLXRType.captionLarge)
                    .foregroundStyle(MLXRColor.textSecondary)
                    .lineLimit(2)

                HStack(spacing: MLXRSpacing.xs) {
                    StatusPill(label: "\(project.assetCount) assets", tint: MLXRColor.brandSecondary)
                    if project.runGroupCount > 0 {
                        StatusPill(label: "\(project.runGroupCount) sets", tint: MLXRColor.brandWarm)
                    }
                    if project.videoCount > 0 {
                        StatusPill(label: "\(project.videoCount) video", tint: MLXRColor.brandPrimary)
                    }
                }
            }

            Spacer(minLength: 0)

            if let onSetCover, let selectedAsset {
                VStack(alignment: .trailing, spacing: MLXRSpacing.xxs) {
                    Button("Set cover", action: onSetCover)
                        .buttonStyle(.bordered)

                    Text(selectedAsset.displayTitle)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textTertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.horizontal, MLXRSpacing.lg)
        .padding(.top, MLXRSpacing.sm)
        .padding(.bottom, MLXRSpacing.sm)
    }

    @ViewBuilder
    private var projectCover: some View {
        if let heroAsset = project.heroAsset {
            LibraryAssetThumbnailView(asset: heroAsset, onMaterialize: onMaterialize)
                .frame(width: 72, height: 72)
                .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
        } else {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(MLXRColor.surfaceCard)
                .frame(width: 72, height: 72)
                .overlay {
                    Image(systemName: "sparkles.rectangle.stack")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundStyle(MLXRColor.brandPrimary)
                }
        }
    }
}
