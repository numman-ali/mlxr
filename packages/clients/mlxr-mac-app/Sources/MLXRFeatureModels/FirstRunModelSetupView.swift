import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

public struct FirstRunModelSetupView: View {
    private let models: [ModelCatalogItem]
    private let previews: [String: SupportedModelPreview]
    private let onLoadPreview: @Sendable (String) async -> Void
    private let onQueueInstall: @Sendable (String) async -> Void
    private let onComplete: () -> Void

    @State private var selectedModelIds: Set<String>
    @State private var isQueueing = false

    public init(
        models: [ModelCatalogItem],
        previews: [String: SupportedModelPreview],
        onLoadPreview: @escaping @Sendable (String) async -> Void,
        onQueueInstall: @escaping @Sendable (String) async -> Void,
        onComplete: @escaping () -> Void
    ) {
        self.models = models
        self.previews = previews
        self.onLoadPreview = onLoadPreview
        self.onQueueInstall = onQueueInstall
        self.onComplete = onComplete
        _selectedModelIds = State(initialValue: Set(models.map(\.modelId)))
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            FeatureHeader(
                eyebrow: "First Run",
                title: "Choose your starting models",
                subtitle: "Pick the recommended rows you want MLXR to queue now. You can always add or remove models later from the Models screen."
            )

            ForEach(models) { item in
                Toggle(isOn: Binding(
                    get: { selectedModelIds.contains(item.modelId) },
                    set: { isOn in
                        if isOn {
                            selectedModelIds.insert(item.modelId)
                        } else {
                            selectedModelIds.remove(item.modelId)
                        }
                    }
                )) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.displayName)
                            .font(.system(.headline, design: .rounded, weight: .semibold))
                        Text(previewSummary(for: item))
                            .foregroundStyle(.secondary)
                    }
                }
                .toggleStyle(.checkbox)
                .task {
                    await onLoadPreview(item.modelId)
                }
            }

            HStack(spacing: 12) {
                Button("Skip for now") {
                    onComplete()
                }
                .buttonStyle(.bordered)

                Button("Queue selected installs") {
                    Task {
                        isQueueing = true
                        for modelId in models.map(\.modelId) where selectedModelIds.contains(modelId) {
                            await onQueueInstall(modelId)
                        }
                        isQueueing = false
                        onComplete()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedModelIds.isEmpty || isQueueing)

                if isQueueing {
                    IndeterminateProgress(label: "Queueing")
                }
            }
        }
        .padding(28)
        .frame(minWidth: 620, minHeight: 420)
    }

    private func previewSummary(for item: ModelCatalogItem) -> String {
        guard let preview = previews[item.modelId] else {
            return item.sourceSummary ?? item.family
        }
        let size = preview.totalSourceBytes.map {
            ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .file)
        } ?? "Unknown size"
        let auth = preview.authRequired ? "login required" : "public"
        return "\(size) • \(auth)"
    }
}
