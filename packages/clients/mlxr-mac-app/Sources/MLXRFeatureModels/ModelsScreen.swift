import AppKit
import MLXRAppDomain
import MLXRDesignSystem
import MLXRRuntimeBridge
import SwiftUI

public struct ModelsScreen: View {
    private let runtimeStatus: RuntimeStatusSnapshot?
    private let catalog: CatalogSnapshot
    private let installOperations: [ModelInstallOperationRecord]
    private let previews: [String: SupportedModelPreview]
    private let installedDetails: [String: InstalledModelDetails]
    private let error: String?
    private let onDismissError: () -> Void
    private let onLoadPreview: @Sendable (String) async -> Void
    private let onInstall: @Sendable (String) async -> Void
    private let onCancelInstall: @Sendable (String) async -> Void
    private let onLoadDetails: @Sendable (String) async -> Void
    private let onRemove: @Sendable (String) async -> Void

    @State private var selectedInstalledModelId: String?

    public init(
        runtimeStatus: RuntimeStatusSnapshot?,
        catalog: CatalogSnapshot,
        installOperations: [ModelInstallOperationRecord],
        previews: [String: SupportedModelPreview],
        installedDetails: [String: InstalledModelDetails],
        error: String?,
        onDismissError: @escaping () -> Void,
        onLoadPreview: @escaping @Sendable (String) async -> Void,
        onInstall: @escaping @Sendable (String) async -> Void,
        onCancelInstall: @escaping @Sendable (String) async -> Void,
        onLoadDetails: @escaping @Sendable (String) async -> Void,
        onRemove: @escaping @Sendable (String) async -> Void
    ) {
        self.runtimeStatus = runtimeStatus
        self.catalog = catalog
        self.installOperations = installOperations
        self.previews = previews
        self.installedDetails = installedDetails
        self.error = error
        self.onDismissError = onDismissError
        self.onLoadPreview = onLoadPreview
        self.onInstall = onInstall
        self.onCancelInstall = onCancelInstall
        self.onLoadDetails = onLoadDetails
        self.onRemove = onRemove
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if let error {
                    InlineErrorBanner(error, onDismiss: onDismissError)
                }

                HeroBanner(
                    title: "Models live here",
                    subtitle: "Choose recommended rows, watch installs move through the queue, and inspect what is already installed in MLXR."
                ) {
                    HStack(spacing: 10) {
                        MetricBadge(label: "Installed", value: "\(catalog.installedItems.count)")
                        MetricBadge(label: "Queue", value: "\(activeOperations.count)")
                    }
                }

                FeatureHeader(
                    eyebrow: "Models",
                    title: "Install, inspect, and remove",
                    subtitle: "Hugging Face is the source. MLXR is the installed home the app manages for you."
                )

                availableSection
                installQueueSection
                installedSection
            }
            .padding(28)
        }
        .sheet(
            isPresented: Binding(
                get: { selectedInstalledModelId != nil },
                set: { isPresented in
                    if !isPresented {
                        selectedInstalledModelId = nil
                    }
                }
            )
        ) {
            if let modelId = selectedInstalledModelId {
                modelDetailSheet(modelId: modelId)
                    .task {
                        await onLoadDetails(modelId)
                    }
            }
        }
    }

    private var availableSection: some View {
        SectionCard(
            title: "Available",
            subtitle: "Recommended rows are ready to queue. Advanced rows stay visible without pretending they are the default."
        ) {
            if availableModels.isEmpty {
                EmptyStateView(
                    title: "Everything visible is already installed",
                    subtitle: "New installs will show up here when the runtime reports more supported rows.",
                    systemImage: "shippingbox"
                )
            } else {
                ForEach(availableModels) { item in
                    availableRow(item)
                        .task {
                            await onLoadPreview(item.modelId)
                        }
                }
            }
        }
    }

    private func availableRow(_ item: ModelCatalogItem) -> some View {
        let operation = operationByModelId[item.modelId]
        let preview = previews[item.modelId]
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.displayName)
                        .font(.system(.headline, design: .rounded, weight: .semibold))
                    Text(item.sourceSummary ?? item.family)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        StatusPill(
                            label: item.isRecommended ? "Recommended" : "Advanced",
                            tint: item.isRecommended ? MLXRTheme.secondaryAccent : MLXRTheme.warmAccent
                        )
                        if let operation {
                            StatusPill(label: phaseLabel(operation.phase), tint: tint(for: operation.phase))
                        } else if let preview, preview.authRequired {
                            StatusPill(label: "Requires login", tint: MLXRTheme.warmAccent)
                        }
                    }
                }
                Spacer()
                if let operation, !operation.phase.isTerminal {
                    Button("Cancel") {
                        Task { await onCancelInstall(operation.operationId) }
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Queue install") {
                        Task { await onInstall(item.modelId) }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            if let preview {
                HStack(spacing: 18) {
                    DetailRow(label: "Download", value: bytesString(preview.totalSourceBytes))
                    DetailRow(label: "Access", value: preview.authRequired ? "Login required" : preview.supportedModel.accessState.capitalized)
                    DetailRow(label: "Tasks", value: item.tasks.map(\.title).joined(separator: ", "))
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var installQueueSection: some View {
        SectionCard(
            title: "Install queue",
            subtitle: "Installs are runtime-owned and serial, so they don’t hijack image or video generation UI."
        ) {
            if installOperations.isEmpty {
                EmptyStateView(
                    title: "No install activity yet",
                    subtitle: "Queued and recent model installs will appear here with their current phase.",
                    systemImage: "clock.arrow.circlepath"
                )
            } else {
                ForEach(installOperations.prefix(8)) { operation in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(operation.supportedModel.displayName)
                                .font(.system(.headline, design: .rounded, weight: .semibold))
                            Spacer()
                            StatusPill(label: phaseLabel(operation.phase), tint: tint(for: operation.phase))
                        }
                        if let error = operation.error {
                            Text(error)
                                .font(.system(.subheadline, design: .rounded))
                                .foregroundStyle(.secondary)
                        } else if let preview = operation.preview {
                            Text(bytesString(preview.totalSourceBytes))
                                .font(.system(.subheadline, design: .rounded))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private var installedSection: some View {
        SectionCard(
            title: "Installed",
            subtitle: "Use these rows in the generation surfaces, inspect what they unlock, or remove them when you want the disk space back."
        ) {
            if catalog.installedItems.isEmpty {
                EmptyStateView(
                    title: "Nothing installed in MLXR yet",
                    subtitle: "Queue a recommended model above to populate the installed list.",
                    systemImage: "internaldrive"
                )
            } else {
                ForEach(catalog.installedItems) { item in
                    Button {
                        selectedInstalledModelId = item.modelId
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.displayName)
                                    .font(.system(.headline, design: .rounded, weight: .semibold))
                                    .foregroundStyle(.primary)
                                Text(item.tasks.map(\.title).joined(separator: ", "))
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            StatusPill(label: item.statusLabel, tint: MLXRTheme.accent)
                        }
                        .padding(.vertical, 6)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func modelDetailSheet(modelId: String) -> some View {
        let item = catalog.items.first { $0.modelId == modelId }
        let details = installedDetails[modelId]
        let resolvedURL = storageURL(for: details)
        return NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text(item?.displayName ?? modelId)
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                    if let item {
                        HStack(spacing: 10) {
                            StatusPill(label: item.family.uppercased(), tint: MLXRTheme.accent)
                            StatusPill(label: item.isRecommended ? "Recommended" : "Advanced", tint: item.isRecommended ? MLXRTheme.secondaryAccent : MLXRTheme.warmAccent)
                        }
                    }

                    if let details {
                        DetailRow(label: "Tasks", value: item?.tasks.map(\.title).joined(separator: ", ") ?? "Unknown")
                        DetailRow(label: "Managed size", value: bytesString(details.managedSizeBytes))
                        if let storageKey = details.managedStorageKey {
                            DetailRow(label: "Storage key", value: storageKey)
                        }
                        if let resolvedURL {
                            DetailRow(label: "Location", value: resolvedURL.path)
                        }
                        if !details.referencedSourceIds.isEmpty {
                            DetailRow(label: "Source refs", value: details.referencedSourceIds.joined(separator: ", "))
                        }
                    } else {
                        IndeterminateProgress(label: "Loading installed model details")
                    }

                    HStack(spacing: 12) {
                        if let resolvedURL {
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([resolvedURL])
                            }
                            .buttonStyle(.bordered)
                        }
                        Button("Remove model") {
                            Task {
                                await onRemove(modelId)
                                selectedInstalledModelId = nil
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(MLXRTheme.destructive)
                    }
                }
                .padding(24)
            }
            .frame(minWidth: 520, minHeight: 420)
        }
    }

    private var availableModels: [ModelCatalogItem] {
        catalog.items.filter { !$0.installed && $0.installable }
    }

    private var activeOperations: [ModelInstallOperationRecord] {
        installOperations.filter { !$0.phase.isTerminal }
    }

    private var operationByModelId: [String: ModelInstallOperationRecord] {
        installOperations.reduce(into: [:]) { partialResult, operation in
            if partialResult[operation.modelId] == nil {
                partialResult[operation.modelId] = operation
            }
        }
    }

    private func storageURL(for details: InstalledModelDetails?) -> URL? {
        guard
            let runtimeStatus,
            let storageKey = details?.managedStorageKey
        else {
            return nil
        }
        return runtimeStatus.runtimeHome.appending(path: storageKey)
    }

    private func phaseLabel(_ phase: ModelInstallOperationPhase) -> String {
        switch phase {
        case .queued:
            "Queued"
        case .resolving:
            "Resolving"
        case .authRequired:
            "Requires login"
        case .downloading:
            "Downloading"
        case .converting:
            "Converting"
        case .registering:
            "Registering"
        case .completed:
            "Installed"
        case .failed:
            "Failed"
        case .cancelled:
            "Cancelled"
        }
    }

    private func tint(for phase: ModelInstallOperationPhase) -> Color {
        switch phase {
        case .completed:
            MLXRTheme.secondaryAccent
        case .failed, .cancelled:
            MLXRTheme.destructive
        case .authRequired:
            MLXRTheme.warmAccent
        default:
            MLXRTheme.accent
        }
    }

    private func bytesString(_ value: Int?) -> String {
        guard let value else { return "Unknown" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file)
    }
}
