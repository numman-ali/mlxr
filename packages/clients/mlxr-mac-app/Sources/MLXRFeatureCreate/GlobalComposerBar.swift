import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

public struct GlobalComposerBar: View {
    @Binding private var workspace: StudioWorkspaceDraft

    private let mode: TaskCategory
    private let availableModes: [TaskCategory]
    private let subworkflows: [WorkflowPresentationSubworkflow]
    private let qualityOptions: [WorkflowPresentationControlOption]
    private let aspectOptions: [WorkflowPresentationControlOption]
    private let durationOptions: [WorkflowPresentationControlOption]
    private let variationOptions: [Int]
    private let selectedModelName: String?
    private let runtimeStatusLabel: String
    private let isPlanning: Bool
    private let isBusy: Bool
    private let canSubmit: Bool
    private let disabledReason: String?
    private let onSelectMode: (TaskCategory) -> Void
    private let onSelectTask: (ProductTask) -> Void
    private let onSelectQuality: (String) -> Void
    private let onSelectAspect: (String) -> Void
    private let onSelectDuration: (String) -> Void
    private let onSelectVariation: (Int) -> Void
    private let onSubmit: () -> Void

    public init(
        workspace: Binding<StudioWorkspaceDraft>,
        mode: TaskCategory,
        availableModes: [TaskCategory],
        subworkflows: [WorkflowPresentationSubworkflow],
        qualityOptions: [WorkflowPresentationControlOption],
        aspectOptions: [WorkflowPresentationControlOption],
        durationOptions: [WorkflowPresentationControlOption],
        variationOptions: [Int],
        selectedModelName: String?,
        runtimeStatusLabel: String,
        isPlanning: Bool,
        isBusy: Bool,
        canSubmit: Bool,
        disabledReason: String?,
        onSelectMode: @escaping (TaskCategory) -> Void,
        onSelectTask: @escaping (ProductTask) -> Void,
        onSelectQuality: @escaping (String) -> Void,
        onSelectAspect: @escaping (String) -> Void,
        onSelectDuration: @escaping (String) -> Void,
        onSelectVariation: @escaping (Int) -> Void,
        onSubmit: @escaping () -> Void
    ) {
        self._workspace = workspace
        self.mode = mode
        self.availableModes = availableModes
        self.subworkflows = subworkflows
        self.qualityOptions = qualityOptions
        self.aspectOptions = aspectOptions
        self.durationOptions = durationOptions
        self.variationOptions = variationOptions
        self.selectedModelName = selectedModelName
        self.runtimeStatusLabel = runtimeStatusLabel
        self.isPlanning = isPlanning
        self.isBusy = isBusy
        self.canSubmit = canSubmit
        self.disabledReason = disabledReason
        self.onSelectMode = onSelectMode
        self.onSelectTask = onSelectTask
        self.onSelectQuality = onSelectQuality
        self.onSelectAspect = onSelectAspect
        self.onSelectDuration = onSelectDuration
        self.onSelectVariation = onSelectVariation
        self.onSubmit = onSubmit
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            HStack(spacing: MLXRSpacing.sm) {
                Picker("Mode", selection: modeBinding) {
                    ForEach(availableModes, id: \.rawValue) { candidate in
                        Text(candidate == .image ? "Image" : "Video")
                            .tag(candidate)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)

                if !subworkflows.isEmpty {
                    Menu {
                        ForEach(subworkflows) { option in
                            Button(option.label) {
                                if let task = ProductTask(rawValue: option.task) {
                                    onSelectTask(task)
                                }
                            }
                        }
                    } label: {
                        Label(currentTaskLabel, systemImage: "chevron.down")
                            .font(MLXRType.bodySmall)
                    }
                    .menuStyle(.borderlessButton)
                }

                Spacer(minLength: 0)
            }

            TextField(
                "What do you want to make?",
                text: $workspace.prompt,
                axis: .vertical
            )
            .textFieldStyle(.plain)
            .font(MLXRType.bodyLarge)
            .lineLimit(2...5)
            .padding(.horizontal, MLXRSpacing.md)
            .padding(.vertical, MLXRSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(MLXRColor.surfaceCard)
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                    )
            )

            HStack(spacing: MLXRSpacing.sm) {
                if let selectedModelName {
                    StatusPill(label: selectedModelName, tint: MLXRColor.brandSecondary)
                }

                controlPills

                StatusPill(
                    label: runtimeStatusLabel,
                    tint: canSubmit ? MLXRColor.brandPrimary : MLXRColor.brandWarm
                )
                if isPlanning {
                    StatusPill(label: "Planning", tint: MLXRColor.brandWarm)
                }

                Spacer(minLength: 0)

                if let disabledReason, !canSubmit {
                    Text(disabledReason)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.trailing)
                }

                Button(action: onSubmit) {
                    HStack(spacing: MLXRSpacing.xs) {
                        if isBusy {
                            ProgressView()
                                .controlSize(.small)
                        }
                        Text(isBusy ? "Working…" : "Generate")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSubmit)
            }
        }
        .padding(MLXRSpacing.lg)
        .background(
            RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                .fill(MLXRColor.canvasRaised)
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.18), radius: 18, y: 8)
        )
    }

    private var modeBinding: Binding<TaskCategory> {
        Binding(
            get: { mode },
            set: { onSelectMode($0) }
        )
    }

    private var currentTaskLabel: String {
        ProductTask(rawValue: workspace.task.rawValue)?.title ?? workspace.task.title
    }

    @ViewBuilder
    private var controlPills: some View {
        if !aspectOptions.isEmpty {
            ComposerControlPill(
                title: workspace.aspectPreset.rawValue,
                options: aspectOptions.map {
                    ComposerControlPill.Option(value: $0.value, label: $0.label)
                },
                action: { onSelectAspect($0.value) }
            )
        }

        if !qualityOptions.isEmpty {
            ComposerControlPill(
                title: qualityLabel,
                options: qualityOptions.map {
                    ComposerControlPill.Option(value: $0.value, label: $0.label)
                },
                action: { onSelectQuality($0.value) }
            )
        }

        if workspace.task.category == .video, !durationOptions.isEmpty {
            ComposerControlPill(
                title: durationLabel,
                options: durationOptions.map {
                    ComposerControlPill.Option(value: $0.value, label: $0.label)
                },
                action: { onSelectDuration($0.value) }
            )
        }

        if workspace.task.category == .image, !variationOptions.isEmpty {
            ComposerControlPill(
                title: "×\(max(1, workspace.variationCount))",
                options: variationOptions.map {
                    ComposerControlPill.Option(value: "\($0)", label: "×\($0)")
                },
                action: { option in
                    if let count = Int(option.value) {
                        onSelectVariation(count)
                    }
                }
            )
        }
    }

    private var qualityLabel: String {
        switch workspace.qualityPreset {
        case .draft:
            "Draft"
        case .standard:
            "Standard"
        case .cinematic:
            "Cinema"
        }
    }

    private var durationLabel: String {
        switch workspace.durationPreset {
        case .short:
            "4s"
        case .medium:
            "8s"
        case .long:
            "12s"
        }
    }
}
