import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct StudioSidebarView: View {
    let showsWorkflowCard: Bool
    let workflows: [StudioWorkflowOption]
    let allowedReferenceKinds: Set<WorkflowReferenceKind>
    @Binding var selectedTask: ProductTask
    let recentAssets: [LibraryAsset]
    @Binding var selectedAssetId: String?
    @Binding var referenceAssetIds: Set<String>
    let onImport: () -> Void
    let onResetDraft: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                if showsWorkflowCard {
                    GlassCard(
                        title: "What do you want to do?",
                        subtitle: "Start with a verb, not a technical pipeline."
                    ) {
                        ForEach(workflows) { workflow in
                            workflowRow(workflow)
                        }
                    }
                }

                GlassCard(
                    title: "Source assets",
                    subtitle: "Choose something you already made or bring in a file from Finder."
                ) {
                    HStack(spacing: MLXRSpacing.sm) {
                        Button("Import from Finder", action: onImport)
                            .buttonStyle(.borderedProminent)
                        Button("New Draft", action: onResetDraft)
                            .buttonStyle(.bordered)
                    }

                    ForEach(recentAssets.prefix(10)) { asset in
                        assetRow(asset)
                    }
                }
            }
            .padding(.horizontal, MLXRSpacing.lg)
            .padding(.vertical, MLXRSpacing.xl)
        }
    }

    @ViewBuilder
    private func workflowRow(_ workflow: StudioWorkflowOption) -> some View {
        let isSelected = workflowMatchesSelection(workflow)
        Button {
            guard case let .available(task) = workflow.availability else { return }
            selectedTask = task
        } label: {
            HStack(spacing: MLXRSpacing.sm) {
                Image(systemName: workflow.icon)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(isSelected ? MLXRColor.brandPrimary : MLXRColor.textSecondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(workflow.title)
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textPrimary)
                    Text(workflow.subtitle)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
                Spacer()
                if case .comingSoon = workflow.availability {
                    StatusPill(label: "Soon", tint: MLXRColor.brandWarm)
                } else if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(MLXRColor.brandPrimary)
                }
            }
            .padding(.vertical, MLXRSpacing.xs)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled({
            if case .comingSoon = workflow.availability { return true }
            return false
        }())
        .opacity({
            if case .comingSoon = workflow.availability { return 0.55 }
            return 1
        }())
    }

    private func assetRow(_ asset: LibraryAsset) -> some View {
        let isSelected = selectedAssetId == asset.id
        let isReference = referenceAssetIds.contains(asset.id)
        let canReference = supportsReference(asset)
        return HStack(spacing: MLXRSpacing.sm) {
            Button {
                selectedAssetId = asset.id
            } label: {
                HStack(spacing: MLXRSpacing.sm) {
                    Image(systemName: icon(for: asset))
                        .foregroundStyle(isSelected ? MLXRColor.brandPrimary : MLXRColor.textSecondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(asset.displayTitle)
                            .font(MLXRType.bodySmall)
                            .foregroundStyle(MLXRColor.textPrimary)
                            .lineLimit(1)
                        Text(asset.sourceSummary)
                            .font(MLXRType.captionLarge)
                            .foregroundStyle(MLXRColor.textTertiary)
                            .lineLimit(1)
                    }
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                if isReference {
                    referenceAssetIds.remove(asset.id)
                } else if canReference {
                    referenceAssetIds.insert(asset.id)
                }
            } label: {
                Image(systemName: isReference ? "checkmark.circle.fill" : "plus.circle")
                    .foregroundStyle(
                        isReference
                            ? MLXRColor.brandSecondary
                            : canReference ? MLXRColor.textTertiary : MLXRColor.textDisabled
                    )
            }
            .buttonStyle(.plain)
            .disabled(!isReference && !canReference)
        }
        .padding(.vertical, MLXRSpacing.xxs)
    }

    private func workflowMatchesSelection(_ workflow: StudioWorkflowOption) -> Bool {
        guard case let .available(task) = workflow.availability else { return false }
        return task == selectedTask
    }

    private func icon(for asset: LibraryAsset) -> String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }

    private func supportsReference(_ asset: LibraryAsset) -> Bool {
        guard let kind = asset.referenceKind else {
            return false
        }
        return allowedReferenceKinds.contains(kind)
    }
}
