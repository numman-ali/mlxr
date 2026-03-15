import MLXRAppDomain
import MLXRDesignSystem
import MLXRRuntimeBridge
@preconcurrency import SwiftUI

public struct SettingsScreen: View {
    private let runtimeStatus: RuntimeStatusSnapshot?
    private let bootstrapError: String?
    private let error: String?
    private let onDismissError: () -> Void
    private let onRefresh: @Sendable () async -> Void
    private let onInspectImport: @Sendable (AdvancedImportDraft) async throws -> [String: SourceInspectionResult]
    private let onRunImport: @Sendable (AdvancedImportDraft) async throws -> Void

    @Binding private var promptHelperMode: PromptHelperMode

    @State private var draft = AdvancedImportDraft()
    @State private var inspectionResults: [String: SourceInspectionResult] = [:]
    @State private var importStatusMessage: String?
    @State private var isImportError: Bool = false
    @State private var isWorking = false

    public init(
        runtimeStatus: RuntimeStatusSnapshot?,
        bootstrapError: String?,
        error: String?,
        onDismissError: @escaping () -> Void,
        promptHelperMode: Binding<PromptHelperMode>,
        onRefresh: @escaping @Sendable () async -> Void,
        onInspectImport: @escaping @Sendable (AdvancedImportDraft) async throws -> [String: SourceInspectionResult],
        onRunImport: @escaping @Sendable (AdvancedImportDraft) async throws -> Void
    ) {
        self.runtimeStatus = runtimeStatus
        self.bootstrapError = bootstrapError
        self.error = error
        self.onDismissError = onDismissError
        _promptHelperMode = promptHelperMode
        self.onRefresh = onRefresh
        self.onInspectImport = onInspectImport
        self.onRunImport = onRunImport
    }

    public var body: some View {
        ZStack {
            AdaptiveBackground()

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                    CompactPageHeader(
                        title: "Settings",
                        subtitle: "Runtime status, prompt-helper defaults, and the advanced import path live here without turning this screen into a second control center."
                    ) {
                        EmptyView()
                    }

                    if let error {
                        InlineErrorBanner(error, onDismiss: onDismissError)
                            .animation(MLXRMotion.snappy, value: error)
                    }

                    if let bootstrapError {
                        InlineErrorBanner(bootstrapError)
                            .animation(MLXRMotion.snappy, value: bootstrapError)
                    }

                    runtimeSection
                    promptHelperSection
                    advancedImportSection
                }
                .padding(.horizontal, MLXRSpacing.xl)
                .padding(.top, MLXRSpacing.lg)
                .padding(.bottom, MLXRSpacing.xl)
            }
        }
    }

    // MARK: - Runtime

    private var runtimeSection: some View {
        DenseSectionSurface {
            sectionLead(
                title: "Runtime",
                subtitle: "The app stays a client. The Python runtime and daemon remain the source of truth, while install and removal work lives in Models."
            )

            if let runtimeStatus {
                HStack(spacing: MLXRSpacing.sm) {
                    StatusPill(
                        label: runtimeStatus.health.status,
                        tint: runtimeStatus.health.status == "ok"
                            ? MLXRColor.brandSecondary
                            : MLXRColor.brandDanger
                    )
                    if let socketPath = runtimeStatus.socketPath {
                        StatusPill(
                            label: socketPath.lastPathComponent,
                            tint: MLXRColor.brandPrimary
                        )
                    }
                }

                DetailRow(label: "Runtime home", value: runtimeStatus.runtimeHome.path)
                DetailRow(label: "Log file", value: runtimeStatus.logFile.path)
            } else {
                Text("Runtime status has not been loaded yet.")
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)
            }

            Button {
                Task { await onRefresh() }
            } label: {
                HStack(spacing: MLXRSpacing.xs) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 13, weight: .semibold))
                    Text("Refresh runtime state")
                        .font(MLXRType.bodySmall)
                }
                .foregroundStyle(.white)
                .padding(.horizontal, MLXRSpacing.lg)
                .padding(.vertical, MLXRSpacing.xs)
                .background(
                    Capsule(style: .continuous)
                        .fill(MLXRColor.brandGradient)
                        .shadow(color: MLXRColor.brandPrimary.opacity(0.25), radius: 8, y: 2)
                )
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Prompt Helper

    private var promptHelperSection: some View {
        DenseSectionSurface {
            sectionLead(
                title: "Prompt helper",
                subtitle: "Prompt enhancement belongs in the app, not in the core runtime contract."
            )

            CreativeControlSegment(
                label: "Mode",
                selection: $promptHelperMode,
                options: PromptHelperMode.allCases.map { ($0, $0.rawValue.capitalized) }
            )

            Text("The current app keeps this setting ready without making core generation depend on it.")
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textTertiary)
        }
    }

    // MARK: - Advanced Import

    private var advancedImportSection: some View {
        DenseSectionSurface {
            sectionLead(
                title: "Advanced import",
                subtitle: "Use the existing runtime inspect, register, and convert routes rather than inventing a parallel app-only model pipeline."
            )

            CreativeControlSegment(
                label: "Mode",
                selection: $draft.mode,
                options: AdvancedImportMode.allCases.map { ($0, $0.title) }
            )

            MLXRTextField("Model ID", text: $draft.modelId, placeholder: "e.g. my-custom-model")
            MLXRTextField("Family hint", text: Binding(
                get: { draft.familyHint ?? "" },
                set: { draft.familyHint = $0.isEmpty ? nil : $0 }
            ), placeholder: "Optional")

            modeSpecificFields

            HStack(spacing: MLXRSpacing.sm) {
                Button {
                    Task { await inspect() }
                } label: {
                    HStack(spacing: MLXRSpacing.xxs) {
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 12, weight: .semibold))
                        Text("Inspect sources")
                            .font(MLXRType.bodySmall)
                    }
                    .foregroundStyle(MLXRColor.brandPrimary)
                    .padding(.horizontal, MLXRSpacing.md)
                    .padding(.vertical, MLXRSpacing.xs)
                    .background(
                        Capsule(style: .continuous)
                            .fill(MLXRColor.brandGlow)
                    )
                }
                .buttonStyle(.plain)
                .disabled(isWorking)

                Button {
                    Task { await runImport() }
                } label: {
                    HStack(spacing: MLXRSpacing.xxs) {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 12, weight: .semibold))
                        Text("Import model")
                            .font(MLXRType.bodySmall)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, MLXRSpacing.md)
                    .padding(.vertical, MLXRSpacing.xs)
                    .background(
                        Capsule(style: .continuous)
                            .fill(MLXRColor.brandGradient)
                            .shadow(color: MLXRColor.brandPrimary.opacity(0.25), radius: 8, y: 2)
                    )
                }
                .buttonStyle(.plain)
                .disabled(isWorking)

                if isWorking {
                    IndeterminateProgress()
                }
            }

            if let importStatusMessage {
                if isImportError {
                    InlineErrorBanner(importStatusMessage)
                } else {
                    InlineSuccessBanner(importStatusMessage)
                }
            }

            if !inspectionResults.isEmpty {
                inspectionResultsView
            }
        }
    }

    private func sectionLead(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            Text(title)
                .font(MLXRType.titleSmall)
                .foregroundStyle(MLXRColor.textPrimary)
            Text(subtitle)
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textSecondary)
        }
    }

    @ViewBuilder
    private var modeSpecificFields: some View {
        switch draft.mode {
        case .huggingFaceSingleRepo:
            MLXRTextField("Hugging Face repo", text: $draft.huggingFaceRepo, placeholder: "org/model-name")
        case .trustedLocalBundle:
            MLXRTextField("Local bundle path", text: $draft.localPath, placeholder: "/path/to/bundle")
        case .ltxMultiSource:
            MLXRTextField("Checkpoint repo", text: $draft.ltxCheckpointRepo, placeholder: "org/checkpoint")
            MLXRTextField("Spatial upsampler repo", text: $draft.ltxUpsamplerRepo, placeholder: "Optional")
            MLXRTextField("Text encoder repo", text: $draft.ltxTextEncoderRepo, placeholder: "org/encoder")
            MLXRTextField("Distilled LoRA repo", text: $draft.ltxDistilledLoRARepo, placeholder: "Optional")
        }
    }

    private var inspectionResultsView: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.md) {
            Text("Inspection results")
                .font(MLXRType.titleMedium)
                .foregroundStyle(MLXRColor.textPrimary)
            ForEach(inspectionResults.keys.sorted(), id: \.self) { role in
                if let result = inspectionResults[role] {
                    VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                        Text(role)
                            .font(MLXRType.titleSmall)
                            .foregroundStyle(MLXRColor.textPrimary)
                        DetailRow(label: "Provider", value: result.resolvedSource.provider)
                        DetailRow(label: "Family", value: result.familyInspection?.family ?? "Unknown")
                        DetailRow(label: "Variant", value: result.familyInspection?.variant ?? "Unknown")
                        DetailRow(label: "Access", value: result.resolvedSource.accessState)
                        DetailRow(label: "License", value: result.resolvedSource.license ?? "Unknown")
                        Text(result.familyInspection?.tasks.joined(separator: ", ") ?? "No tasks detected")
                            .font(MLXRType.bodySmall)
                            .foregroundStyle(MLXRColor.textTertiary)
                    }
                    .padding(MLXRSpacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .fill(Color.white.opacity(0.05))
                            .overlay(
                                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                                    .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                            )
                    )
                }
            }
        }
    }

    // MARK: - Helpers
    private func inspect() async {
        isWorking = true
        defer { isWorking = false }
        do {
            inspectionResults = try await onInspectImport(draft)
            isImportError = false
            importStatusMessage = "Inspection complete."
        } catch {
            isImportError = true
            importStatusMessage = "Inspection failed: \(error.localizedDescription)"
        }
    }

    private func runImport() async {
        isWorking = true
        defer { isWorking = false }
        do {
            try await onRunImport(draft)
            isImportError = false
            importStatusMessage = "Import complete."
        } catch {
            isImportError = true
            importStatusMessage = "Import failed: \(error.localizedDescription)"
        }
    }
}
