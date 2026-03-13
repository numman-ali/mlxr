import MLXRAppDomain
import MLXRDesignSystem
import MLXRRuntimeBridge
import SwiftUI

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
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                FeatureHeader(
                    eyebrow: "Settings",
                    title: "Runtime, models, and import",
                    subtitle: "This is where the app stays honest about runtime state, installs curated defaults, and opens the advanced import path without pretending everything is equally ready."
                )

                if let error {
                    InlineErrorBanner(error, onDismiss: onDismissError)
                        .animation(MLXRAnimation.snappy, value: error)
                }

                if let bootstrapError {
                    InlineErrorBanner(bootstrapError)
                        .animation(MLXRAnimation.snappy, value: bootstrapError)
                }

                runtimeSection
                promptHelperSection
                advancedImportSection
            }
            .padding(28)
        }
    }

    // MARK: - Runtime

    private var runtimeSection: some View {
        SectionCard(
            title: "Runtime",
            subtitle: "The app is a client. The Python runtime and daemon stay the source of truth, while model installs and removals now live in the dedicated Models screen."
        ) {
            if let runtimeStatus {
                HStack(spacing: 10) {
                    StatusPill(
                        label: runtimeStatus.health.status,
                        tint: runtimeStatus.health.status == "ok"
                            ? MLXRTheme.secondaryAccent
                            : MLXRTheme.destructive
                    )
                    if let socketPath = runtimeStatus.socketPath {
                        StatusPill(
                            label: socketPath.lastPathComponent,
                            tint: MLXRTheme.accent
                        )
                    }
                }

                DetailRow(label: "Runtime home", value: runtimeStatus.runtimeHome.path)
                DetailRow(label: "Log file", value: runtimeStatus.logFile.path)
            } else {
                Text("Runtime status has not been loaded yet.")
                    .foregroundStyle(.secondary)
            }

            Button("Refresh runtime state") {
                Task { await onRefresh() }
            }
            .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Prompt Helper

    private var promptHelperSection: some View {
        SectionCard(
            title: "Prompt helper",
            subtitle: "Prompt enhancement belongs in the app, not in the core runtime contract. This is a placeholder for the first local helper track."
        ) {
            Picker("Mode", selection: $promptHelperMode) {
                ForEach(PromptHelperMode.allCases) { mode in
                    Text(mode.rawValue.capitalized).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text("The current app keeps this setting ready without making core generation depend on it.")
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Advanced Import

    private var advancedImportSection: some View {
        SectionCard(
            title: "Advanced import",
            subtitle: "Use the existing runtime source inspect, register, and convert routes rather than inventing a parallel app-only model pipeline."
        ) {
            Picker("Mode", selection: $draft.mode) {
                ForEach(AdvancedImportMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            TextField("Model ID", text: $draft.modelId)
                .textFieldStyle(.roundedBorder)
            TextField("Family hint (optional)", text: Binding(
                get: { draft.familyHint ?? "" },
                set: { draft.familyHint = $0.isEmpty ? nil : $0 }
            ))
            .textFieldStyle(.roundedBorder)

            modeSpecificFields

            HStack {
                Button("Inspect sources") {
                    Task { await inspect() }
                }
                .buttonStyle(.bordered)
                .disabled(isWorking)

                Button("Import model") {
                    Task { await runImport() }
                }
                .buttonStyle(.borderedProminent)
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

    @ViewBuilder
    private var modeSpecificFields: some View {
        switch draft.mode {
        case .huggingFaceSingleRepo:
            TextField("Hugging Face repo", text: $draft.huggingFaceRepo)
                .textFieldStyle(.roundedBorder)
        case .trustedLocalBundle:
            TextField("Local bundle path", text: $draft.localPath)
                .textFieldStyle(.roundedBorder)
        case .ltxMultiSource:
            TextField("Checkpoint repo", text: $draft.ltxCheckpointRepo)
                .textFieldStyle(.roundedBorder)
            TextField("Spatial upsampler repo (optional)", text: $draft.ltxUpsamplerRepo)
                .textFieldStyle(.roundedBorder)
            TextField("Text encoder repo", text: $draft.ltxTextEncoderRepo)
                .textFieldStyle(.roundedBorder)
            TextField("Distilled LoRA repo (optional)", text: $draft.ltxDistilledLoRARepo)
                .textFieldStyle(.roundedBorder)
        }
    }

    private var inspectionResultsView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Inspection results")
                .font(.system(.headline, design: .rounded, weight: .semibold))
            ForEach(inspectionResults.keys.sorted(), id: \.self) { role in
                if let result = inspectionResults[role] {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(role)
                            .font(.system(.headline, design: .rounded, weight: .semibold))
                        DetailRow(label: "Provider", value: result.resolvedSource.provider)
                        DetailRow(label: "Family", value: result.familyInspection?.family ?? "Unknown")
                        DetailRow(label: "Variant", value: result.familyInspection?.variant ?? "Unknown")
                        DetailRow(label: "Access", value: result.resolvedSource.accessState)
                        DetailRow(label: "License", value: result.resolvedSource.license ?? "Unknown")
                        Text(result.familyInspection?.tasks.joined(separator: ", ") ?? "No tasks detected")
                            .foregroundStyle(.secondary)
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(MLXRTheme.surfaceSecondary)
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
