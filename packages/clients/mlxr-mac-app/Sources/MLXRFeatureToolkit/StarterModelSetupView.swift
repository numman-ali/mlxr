import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

@MainActor
public struct StarterModelSetupView: View {
    let catalog: CatalogSnapshot
    let previews: [String: SupportedModelPreview]
    let onLoadPreview: @Sendable (String) async -> Void
    let onQueueInstall: @Sendable (String) async -> Void
    let onComplete: () -> Void
    let onSkip: () -> Void

    @State private var selectedModelIds: Set<String> = []

    public init(
        catalog: CatalogSnapshot,
        previews: [String: SupportedModelPreview],
        onLoadPreview: @escaping @Sendable (String) async -> Void,
        onQueueInstall: @escaping @Sendable (String) async -> Void,
        onComplete: @escaping () -> Void,
        onSkip: @escaping () -> Void
    ) {
        self.catalog = catalog
        self.previews = previews
        self.onLoadPreview = onLoadPreview
        self.onQueueInstall = onQueueInstall
        self.onComplete = onComplete
        self.onSkip = onSkip
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xl) {
            FeatureHeader(
                eyebrow: "First Run",
                title: "Pick your starter models",
                subtitle: "Install the rows you want once, then use them across your projects and Library."
            )

            ScrollView {
                VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                    ForEach(catalog.recommendedAvailableItems) { item in
                        modelCard(item)
                            .task { await onLoadPreview(item.modelId) }
                    }
                }
            }

            HStack {
                Button("Skip for now", action: onSkip)
                    .buttonStyle(.bordered)
                Spacer()
                Button("Queue selected installs") {
                    Task {
                        for modelId in selectedModelIds.sorted() {
                            await onQueueInstall(modelId)
                        }
                        onComplete()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedModelIds.isEmpty)
            }
        }
        .padding(MLXRSpacing.xl)
        .frame(minWidth: 760, minHeight: 540)
        .background(AdaptiveBackground())
        .task {
            if selectedModelIds.isEmpty {
                selectedModelIds = Set(catalog.recommendedAvailableItems.prefix(2).map(\.modelId))
            }
        }
    }

    private func modelCard(_ item: ModelCatalogItem) -> some View {
        let isSelected = selectedModelIds.contains(item.modelId)
        let preview = previews[item.modelId]
        return Button {
            if isSelected {
                selectedModelIds.remove(item.modelId)
            } else {
                selectedModelIds.insert(item.modelId)
            }
        } label: {
            GlassCardInteractive {
                VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.displayName)
                                .font(MLXRType.titleSmall)
                                .foregroundStyle(MLXRColor.textPrimary)
                            Text(item.tasks.map(\.title).joined(separator: ", "))
                                .font(MLXRType.bodySmall)
                                .foregroundStyle(MLXRColor.textSecondary)
                        }
                        Spacer()
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(isSelected ? MLXRColor.brandPrimary : MLXRColor.textTertiary)
                    }

                    HStack(spacing: MLXRSpacing.xs) {
                        StatusPill(label: item.family.uppercased(), tint: MLXRColor.brandPrimary)
                        if let preview, preview.authRequired {
                            StatusPill(label: "Requires login", tint: MLXRColor.brandWarm)
                        }
                    }

                    if let preview {
                        Text("Download: \(bytesString(preview.totalSourceBytes))")
                            .font(MLXRType.captionLarge)
                            .foregroundStyle(MLXRColor.textTertiary)
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }

    private func bytesString(_ value: Int?) -> String {
        guard let value else { return "Unknown size" }
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: Int64(value))
    }
}
