import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct StudioInspectorView: View {
    @Binding var workspace: StudioWorkspaceDraft
    let selectedModel: ModelCatalogItem?
    let availableModels: [ModelCatalogItem]
    let availablePacks: [PackRecord]
    let selectedPack: PackRecord?
    let installOperation: ModelInstallOperationRecord?
    let resolvedSettings: StudioResolvedSettings
    let shouldShowResetToRecommended: Bool
    let referenceAssets: [LibraryAsset]
    let referenceRequirements: [WorkflowReferenceRequirement]
    let readinessWarnings: [String]
    let isPlanning: Bool
    let onSelectModel: (String) -> Void
    let onSelectPack: (String?) -> Void
    let onQueueInstall: @Sendable (String) async -> Void
    let onResetToRecommended: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                modelCard
                packCard
                presetsCard
                referencesCard
                advancedCard
            }
            .padding(.horizontal, MLXRSpacing.lg)
            .padding(.vertical, MLXRSpacing.xl)
        }
    }

    private var modelCard: some View {
        GlassCard(
            title: "Model",
            subtitle: "Pick a row that supports this workflow."
        ) {
            if availableModels.isEmpty {
                EmptyStateView(
                    title: "No model available",
                    subtitle: "Install a compatible row in Models before running this workflow.",
                    systemImage: "shippingbox"
                )
            } else {
                Picker("Model", selection: Binding(
                    get: { selectedModel?.modelId ?? "" },
                    set: { onSelectModel($0) }
                )) {
                    ForEach(availableModels) { model in
                        Text(model.displayName).tag(model.modelId)
                    }
                }
                .pickerStyle(.menu)

                if let selectedModel {
                    HStack(spacing: MLXRSpacing.xs) {
                        StatusPill(label: selectedModel.sectionLabel, tint: selectedModel.isRecommended ? MLXRColor.brandSecondary : MLXRColor.brandWarm)
                        StatusPill(label: selectedModel.statusLabel, tint: MLXRColor.brandPrimary)
                    }
                    if !selectedModel.installed {
                        if let installOperation, !installOperation.phase.isTerminal {
                            StatusPill(label: phaseLabel(installOperation.phase), tint: MLXRColor.brandWarm)
                        } else {
                            Button("Install \(selectedModel.displayName)") {
                                Task { await onQueueInstall(selectedModel.modelId) }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var packCard: some View {
        if !availablePacks.isEmpty {
            GlassCard(
                title: "Look / Pack",
                subtitle: "Friendly wrappers for family-local style, motion, and control options."
            ) {
                Picker("Pack", selection: Binding(
                    get: { selectedPack?.id ?? "" },
                    set: { onSelectPack($0.isEmpty ? nil : $0) }
                )) {
                    ForEach(availablePacks) { pack in
                        Text(pack.title).tag(pack.id)
                    }
                }
                .pickerStyle(.segmented)

                if let selectedPack {
                    Text(selectedPack.summary)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                }
            }
        }
    }

    private var presetsCard: some View {
        GlassCard(
            title: "Presets",
            subtitle: "Use named defaults first. Raw numbers stay tucked away until you ask for them."
        ) {
            CreativeControlSegment(
                label: "Quality",
                selection: $workspace.qualityPreset,
                options: [
                    (.draft, "Draft"),
                    (.standard, "Standard"),
                    (.cinematic, "Cinema"),
                ]
            )
            CreativeControlSegment(
                label: "Size",
                selection: $workspace.aspectPreset,
                options: aspectOptions
            )
            if workspace.task.category == .video {
                CreativeControlSegment(
                    label: "Duration",
                    selection: $workspace.durationPreset,
                    options: DurationPreset.allCases.map { ($0, "\($0.rawValue) \($0.approximateSeconds)") }
                )
            }
            if workspace.task.category == .image {
                CreativeControlSegment(
                    label: "Variations",
                    selection: Binding(
                        get: { workspace.variationCount },
                        set: { workspace.variationCount = $0 }
                    ),
                    options: [(1, "1"), (2, "2"), (4, "4")]
                )
            }

            DetailRow(label: "Resolution", value: "\(resolvedSettings.width) × \(resolvedSettings.height)")
            if workspace.task.category == .video {
                DetailRow(label: "Frames", value: "\(resolvedSettings.numFrames) at \(resolvedSettings.fps) fps")
            }

            if shouldShowResetToRecommended {
                Button("Reset to recommended", action: onResetToRecommended)
                    .buttonStyle(.bordered)
            }
        }
    }

    private var referencesCard: some View {
        GlassCard(
            title: "References",
            subtitle: referenceSubtitle
        ) {
            if isPlanning {
                HStack(spacing: MLXRSpacing.xs) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Checking the current draft against the runtime…")
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                }
            }

            if !referenceRequirements.isEmpty {
                VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                    ForEach(referenceRequirements) { requirement in
                        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                            Text(requirement.description)
                                .font(MLXRType.bodySmall)
                                .foregroundStyle(MLXRColor.textPrimary)
                            Text(requirementSummary(requirement))
                                .font(MLXRType.captionLarge)
                                .foregroundStyle(MLXRColor.textTertiary)
                        }
                    }
                }
            }

            if !readinessWarnings.isEmpty {
                VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                    ForEach(readinessWarnings, id: \.self) { warning in
                        Label {
                            Text(warning)
                                .font(MLXRType.bodySmall)
                                .foregroundStyle(MLXRColor.textSecondary)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(MLXRColor.brandWarm)
                        }
                    }
                }
            }

            if referenceAssets.isEmpty {
                Text("No references selected yet.")
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
            } else {
                ForEach(referenceAssets) { asset in
                    HStack {
                        Image(systemName: icon(for: asset))
                            .foregroundStyle(MLXRColor.brandPrimary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(asset.displayTitle)
                                .font(MLXRType.bodySmall)
                                .foregroundStyle(MLXRColor.textPrimary)
                            Text(asset.sourceSummary)
                                .font(MLXRType.captionLarge)
                                .foregroundStyle(MLXRColor.textTertiary)
                        }
                        Spacer()
                    }
                }
            }
        }
    }

    private var advancedCard: some View {
        GlassCard(
            title: "Advanced",
            subtitle: "Only use this when you need to override the recommended settings for this model."
        ) {
            Toggle("Use custom values", isOn: $workspace.useCustomSettings)
            Toggle("Show advanced controls", isOn: $workspace.showAdvanced)

            if workspace.showAdvanced {
                Group {
                    field("Width", value: $workspace.manualWidth)
                    field("Height", value: $workspace.manualHeight)
                    if workspace.task.category == .video {
                        field("Frames", value: $workspace.manualFrames)
                        field("FPS", value: $workspace.manualFps)
                    }
                    field("Steps", value: $workspace.manualSteps)
                    field("Guidance", value: $workspace.manualGuidance, fractional: true)
                    MLXRTextField("Seed", text: $workspace.seed, placeholder: "Optional")
                    Picker("Format", selection: $workspace.artifactFormat) {
                        if workspace.task.category == .image {
                            Text("PNG").tag("png")
                            Text("JPG").tag("jpg")
                        } else {
                            Text("MP4").tag("mp4")
                        }
                    }
                    .pickerStyle(.segmented)
                }
                .disabled(!workspace.useCustomSettings)
            }
        }
    }

    private var aspectOptions: [(AspectPreset, String)] {
        let base: [AspectPreset] = workspace.task.category == .image
            ? [.square, .landscape, .portrait, .story]
            : [.square, .landscape, .portrait]
        return base.map { ($0, $0.rawValue) }
    }

    private var referenceSubtitle: String {
        if referenceRequirements.isEmpty {
            return "This workflow does not need source material unless the selected model says otherwise."
        }
        return "The runtime plan decides what kinds of source material are valid here."
    }

    private func requirementSummary(_ requirement: WorkflowReferenceRequirement) -> String {
        let minimum = requirement.minimumCount
        let maximum = requirement.maximumCount
        let countSummary: String
        switch (minimum, maximum) {
        case (0, .none):
            countSummary = "Optional"
        case let (min, .none):
            countSummary = "At least \(min)"
        case let (min, .some(max)) where min == max:
            countSummary = "Exactly \(max)"
        case let (min, .some(max)):
            countSummary = "\(min) to \(max)"
        }

        if requirement.acceptedRoles.isEmpty {
            return "\(countSummary) \(requirement.kind.rawValue) reference"
        }
        let roles = requirement.acceptedRoles.joined(separator: ", ")
        return "\(countSummary) \(requirement.kind.rawValue) reference (\(roles))"
    }

    private func phaseLabel(_ phase: ModelInstallOperationPhase) -> String {
        switch phase {
        case .queued: "Queued"
        case .resolving: "Resolving"
        case .authRequired: "Requires login"
        case .downloading: "Downloading"
        case .converting: "Converting"
        case .registering: "Registering"
        case .completed: "Installed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }

    private func icon(for asset: LibraryAsset) -> String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }

    private func field(
        _ label: String,
        value: Binding<Double>,
        fractional: Bool = false
    ) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            if fractional {
                TextField(
                    label,
                    value: value,
                    format: .number.precision(.fractionLength(1))
                )
                .textFieldStyle(.roundedBorder)
            } else {
                TextField(label, value: value, format: .number)
                    .textFieldStyle(.roundedBorder)
            }
        }
    }
}
