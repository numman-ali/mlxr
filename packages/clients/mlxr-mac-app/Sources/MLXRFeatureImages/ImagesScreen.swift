import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI
import UniformTypeIdentifiers

public struct ImagesScreen: View {
    private let catalog: CatalogSnapshot
    private let isBusy: Bool
    private let installOperations: [ModelInstallOperationRecord]
    private let error: String?
    private let onDismissError: () -> Void
    private let onInstall: @Sendable (String) async -> Void
    private let onSubmit: @Sendable (ImageGenerationRequest) async -> Void

    @State private var selectedTask: ProductTask = .imageGenerate
    @State private var selectedModelId = ""
    @State private var prompt = ""
    @State private var negativePrompt = ""
    @State private var width = 1024.0
    @State private var height = 1024.0
    @State private var numInferenceSteps = 8.0
    @State private var guidanceScale = 4.0
    @State private var seed = ""
    @State private var artifactFormat = "png"
    @State private var qwenSchedulerPreset = "default"
    @State private var fluxQuantizeBits = "none"
    @State private var isPickingReferences = false
    @State private var referenceFiles: [MediaReferenceInput] = []

    public init(
        catalog: CatalogSnapshot,
        isBusy: Bool,
        installOperations: [ModelInstallOperationRecord],
        error: String?,
        onDismissError: @escaping () -> Void,
        onInstall: @escaping @Sendable (String) async -> Void,
        onSubmit: @escaping @Sendable (ImageGenerationRequest) async -> Void
    ) {
        self.catalog = catalog
        self.isBusy = isBusy
        self.installOperations = installOperations
        self.error = error
        self.onDismissError = onDismissError
        self.onInstall = onInstall
        self.onSubmit = onSubmit
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if let error {
                    InlineErrorBanner(error, onDismiss: onDismissError)
                        .animation(MLXRAnimation.snappy, value: error)
                }

                HeroBanner(
                    title: "Images that feel local, fast, and real",
                    subtitle: "Generate or edit still images with the same MLXR runtime the CLI uses, while keeping recommended defaults separate from advanced rows."
                ) {
                    HStack(spacing: 10) {
                        StatusPill(label: "Recommended", tint: MLXRTheme.secondaryAccent)
                        StatusPill(label: "Advanced", tint: MLXRTheme.warmAccent)
                    }
                }

                FeatureHeader(
                    eyebrow: "Images",
                    title: selectedTask.title,
                    subtitle: selectedTask.subtitle
                )

                taskSection
                modelSection
                promptSection

                if selectedTask == .imageEdit {
                    referencesSection
                }

                outputSection

                if let selectedModel {
                    advancedSection(for: selectedModel)
                }

                submitBar
            }
            .padding(28)
        }
        .animation(MLXRAnimation.spring, value: selectedTask)
        .onAppear(perform: syncSelection)
        .onChange(of: selectedTask) { _, _ in
            syncSelection()
        }
        .fileImporter(
            isPresented: $isPickingReferences,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result else { return }
            referenceFiles.append(contentsOf: urls.map {
                MediaReferenceInput(fileURL: $0, kind: .image)
            })
        }
    }

    // MARK: - Task

    private var taskSection: some View {
        SectionCard(
            title: "Task",
            subtitle: "Start with the simple flow and opt into editing only when you need references."
        ) {
            Picker("Mode", selection: $selectedTask) {
                Text(ProductTask.imageGenerate.title).tag(ProductTask.imageGenerate)
                Text(ProductTask.imageEdit.title).tag(ProductTask.imageEdit)
            }
            .pickerStyle(.segmented)
        }
    }

    // MARK: - Model

    private var modelSection: some View {
        SectionCard(
            title: "Model",
            subtitle: "Recommended rows stay easy to find. Advanced installed rows remain available without pretending they are the default."
        ) {
            if availableModels.isEmpty {
                EmptyStateView(
                    title: "No image models available yet",
                    subtitle: "Queue a recommended image row in the Models screen or import an advanced row there first.",
                    systemImage: "photo.on.rectangle"
                )
            } else {
                Picker("Model", selection: $selectedModelId) {
                    ForEach(availableModels) { item in
                        Text("\(item.displayName) \u{2022} \(item.sectionLabel)")
                            .tag(item.modelId)
                    }
                }
                .pickerStyle(.menu)

                if let selectedModel {
                    HStack(spacing: 10) {
                        StatusPill(
                            label: selectedModel.sectionLabel,
                            tint: selectedModel.isRecommended
                                ? MLXRTheme.secondaryAccent
                                : MLXRTheme.warmAccent
                        )
                        StatusPill(
                            label: selectedModel.statusLabel,
                            tint: MLXRTheme.accent
                        )
                    }

                    if let sourceSummary = selectedModel.sourceSummary {
                        Text(sourceSummary)
                            .font(.system(.subheadline, design: .rounded))
                            .foregroundStyle(.secondary)
                    }

                    if !selectedModel.installed && selectedModel.installable {
                        if let operation = installOperation(for: selectedModel.modelId), !operation.phase.isTerminal {
                            Button(phaseLabel(operation.phase)) {}
                                .buttonStyle(.bordered)
                                .disabled(true)
                        } else {
                            Button("Queue install for \(selectedModel.displayName)") {
                                Task { await onInstall(selectedModel.modelId) }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Prompt

    private var promptSection: some View {
        SectionCard(
            title: "Prompt",
            subtitle: "Keep the original prompt visible and direct. Negative prompt stays optional."
        ) {
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
            TextField("Negative prompt", text: $negativePrompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
        }
    }

    // MARK: - References

    private var referencesSection: some View {
        SectionCard(
            title: "References",
            subtitle: "Editing supports one or more reference images where the selected model and row allow it."
        ) {
            Button(referenceFiles.isEmpty
                ? "Choose reference images"
                : "Add more reference images"
            ) {
                isPickingReferences = true
            }
            .buttonStyle(.bordered)

            if referenceFiles.isEmpty {
                Text("No references selected yet.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(referenceFiles) { reference in
                    HStack {
                        Image(systemName: "photo")
                        Text(reference.fileURL.lastPathComponent)
                        Spacer()
                        Button("Remove") {
                            referenceFiles.removeAll { $0.id == reference.id }
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
    }

    // MARK: - Output

    private var outputSection: some View {
        SectionCard(
            title: "Output",
            subtitle: "The defaults are meant to be useful, not magical. Everything stays visible and editable."
        ) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                GridRow {
                    LabeledContent("Width") {
                        TextField("Width", value: $width, format: .number)
                            .frame(width: 90)
                    }
                    LabeledContent("Height") {
                        TextField("Height", value: $height, format: .number)
                            .frame(width: 90)
                    }
                    LabeledContent("Steps") {
                        TextField("Steps", value: $numInferenceSteps, format: .number)
                            .frame(width: 90)
                    }
                }
                GridRow {
                    LabeledContent("Guidance") {
                        TextField(
                            "Guidance",
                            value: $guidanceScale,
                            format: .number.precision(.fractionLength(1))
                        )
                        .frame(width: 90)
                    }
                    LabeledContent("Seed") {
                        TextField("Optional", text: $seed)
                            .frame(width: 120)
                    }
                    LabeledContent("Format") {
                        Picker("Format", selection: $artifactFormat) {
                            Text("PNG").tag("png")
                            Text("JPG").tag("jpg")
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 140)
                    }
                }
            }
        }
    }

    // MARK: - Advanced

    private func advancedSection(for model: ModelCatalogItem) -> some View {
        SectionCard(
            title: "Advanced",
            subtitle: "These stay family-local on purpose so the shared UI does not collapse into one giant debug form."
        ) {
            if model.family == "qwen_image" {
                Picker("Scheduler preset", selection: $qwenSchedulerPreset) {
                    Text("Default").tag("default")
                    Text("Lightning").tag("lightning")
                    Text("Wuli").tag("turbo_wuli")
                }
                .pickerStyle(.segmented)
            } else if model.family == "flux2" {
                Picker("Quantization", selection: $fluxQuantizeBits) {
                    Text("Off").tag("none")
                    Text("8-bit").tag("8")
                    Text("6-bit").tag("6")
                    Text("4-bit").tag("4")
                }
                .pickerStyle(.segmented)
            } else {
                Text("This model family does not currently expose extra image-local controls in the first app.")
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Submit

    private var submitBar: some View {
        HStack {
            Button {
                Task { await submit() }
            } label: {
                HStack(spacing: 8) {
                    if isBusy {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Text(selectedTask == .imageGenerate ? "Generate image" : "Edit image")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(
                isBusy
                    || selectedModel == nil
                    || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )

            if let selectedModel {
                Text("\(selectedModel.tasks.count) supported task(s) on this row")
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Helpers

    private var availableModels: [ModelCatalogItem] {
        catalog.items(for: selectedTask)
    }

    private var selectedModel: ModelCatalogItem? {
        availableModels.first(where: { $0.modelId == selectedModelId })
            ?? catalog.defaultModel(for: selectedTask)
    }

    private func syncSelection() {
        if let defaultModel = catalog.defaultModel(for: selectedTask) {
            selectedModelId = defaultModel.modelId
        }
    }

    private func installOperation(for modelId: String) -> ModelInstallOperationRecord? {
        installOperations.first { $0.modelId == modelId }
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
            "Retry install"
        case .cancelled:
            "Cancelled"
        }
    }

    private func submit() async {
        guard let selectedModel else { return }

        var extensions: JSONMap = [:]
        if selectedModel.family == "qwen_image" {
            extensions["scheduler_preset"] = .string(qwenSchedulerPreset)
        } else if selectedModel.family == "flux2",
                  fluxQuantizeBits != "none",
                  let bits = Int(fluxQuantizeBits)
        {
            extensions["quantize_bits"] = .integer(bits)
        }

        await onSubmit(
            ImageGenerationRequest(
                task: selectedTask,
                modelId: selectedModel.modelId,
                prompt: prompt,
                negativePrompt: negativePrompt,
                width: Int(width),
                height: Int(height),
                numInferenceSteps: Int(numInferenceSteps),
                guidanceScale: guidanceScale,
                seed: Int(seed),
                artifactFormat: artifactFormat,
                references: referenceFiles,
                familyExtensions: extensions
            )
        )
    }
}
