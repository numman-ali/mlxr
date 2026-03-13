import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI
import UniformTypeIdentifiers

public struct VideoScreen: View {
    private let catalog: CatalogSnapshot
    private let isBusy: Bool
    private let installOperations: [ModelInstallOperationRecord]
    private let error: String?
    private let onDismissError: () -> Void
    private let onInstall: @Sendable (String) async -> Void
    private let onSubmit: @Sendable (VideoGenerationRequest) async -> Void

    @State private var selectedTask: ProductTask = .videoGenerate
    @State private var selectedModelId = ""
    @State private var prompt = ""
    @State private var negativePrompt = ""
    @State private var width = 704.0
    @State private var height = 704.0
    @State private var numFrames = 81.0
    @State private var fps = 24.0
    @State private var numInferenceSteps = 40.0
    @State private var guidanceScale = 3.0
    @State private var seed = ""
    @State private var workflowVariant = ""
    @State private var controlVariant = "ic_lora"
    @State private var conditioningAttentionStrength = 1.0
    @State private var windowStartSeconds = 0.0
    @State private var windowEndSeconds = 4.0
    @State private var regenerateVideo = true
    @State private var regenerateAudio = true
    @State private var imageReferences: [ImageReferenceDraft] = []
    @State private var selectedVideoURL: URL?
    @State private var selectedAudioURL: URL?
    @State private var selectedLoRAURL: URL?
    @State private var isPickingImages = false
    @State private var isPickingVideo = false
    @State private var isPickingAudio = false
    @State private var isPickingLoRA = false

    public init(
        catalog: CatalogSnapshot,
        isBusy: Bool,
        installOperations: [ModelInstallOperationRecord],
        error: String?,
        onDismissError: @escaping () -> Void,
        onInstall: @escaping @Sendable (String) async -> Void,
        onSubmit: @escaping @Sendable (VideoGenerationRequest) async -> Void
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
            VStack(alignment: .leading, spacing: 20) {
                HeroBanner(
                    title: "Local video workflows, not just one demo path",
                    subtitle: "Text to video, image to video, conditioned audio, video control, interpolation, and retake all ride the same MLXR daemon and job model."
                ) {
                    HStack(spacing: 10) {
                        ForEach(videoTasks) { task in
                            StatusPill(
                                label: task.title,
                                tint: task.isAdvancedByDefault ? MLXRTheme.warmAccent : MLXRTheme.secondaryAccent
                            )
                        }
                    }
                }

                if let error {
                    InlineErrorBanner(error, onDismiss: onDismissError)
                        .transition(.move(edge: .top).combined(with: .opacity))
                        .animation(MLXRAnimation.snappy, value: error)
                }

                FeatureHeader(
                    eyebrow: "Video",
                    title: selectedTask.title,
                    subtitle: selectedTask.subtitle
                )

                // MARK: Task Picker

                SectionCard(title: "Task", subtitle: "Recommended flows stay close at hand. More experimental or unpromoted but real runtime slices remain available under Advanced modes.") {
                    Picker("Task", selection: $selectedTask) {
                        ForEach(videoTasks) { task in
                            Text(task.title).tag(task)
                        }
                    }
                    .pickerStyle(.menu)
                }

                // MARK: Model Picker

                SectionCard(title: "Model", subtitle: "This list is driven by the runtime's actual capability surface, not by a hardcoded app menu.") {
                    if availableModels.isEmpty {
                        EmptyStateView(
                            title: "No video models available yet",
                            subtitle: "Queue LTX in the Models screen or import an advanced video row there first.",
                            systemImage: "film.stack"
                        )
                    } else {
                        Picker("Model", selection: $selectedModelId) {
                            ForEach(availableModels) { item in
                                Text("\(item.displayName) \u{2022} \(item.sectionLabel)").tag(item.modelId)
                            }
                        }
                        .pickerStyle(.menu)

                        if let selectedModel {
                            HStack(spacing: 10) {
                                StatusPill(
                                    label: selectedModel.sectionLabel,
                                    tint: selectedModel.isRecommended ? MLXRTheme.secondaryAccent : MLXRTheme.warmAccent
                                )
                                StatusPill(label: selectedModel.statusLabel, tint: MLXRTheme.accent)
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

                // MARK: Prompt

                SectionCard(title: "Prompt", subtitle: "Prompt stays explicit even for reference-driven tasks so the result remains steerable.") {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                    TextField("Negative prompt", text: $negativePrompt, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                }

                // MARK: Input References

                if needsImageReferences || needsVideoReference || needsAudioReference || needsLoRAReference {
                    SectionCard(title: "Inputs", subtitle: "Each advanced mode maps to the runtime's actual reference kinds and metadata contract.") {
                        if needsImageReferences {
                            Button(imageReferences.isEmpty ? "Choose image references" : "Add image references") {
                                isPickingImages = true
                            }
                            .buttonStyle(.bordered)

                            ForEach(Array(imageReferences.indices), id: \.self) { index in
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Image(systemName: "photo")
                                        Text(imageReferences[index].fileURL.lastPathComponent)
                                            .font(.system(.body, design: .rounded, weight: .medium))
                                        Spacer()
                                        Button("Remove") {
                                            imageReferences.remove(at: index)
                                        }
                                        .buttonStyle(.borderless)
                                    }
                                    HStack {
                                        Stepper(
                                            "Frame \(imageReferences[index].frameIndex)",
                                            value: Binding(
                                                get: { imageReferences[index].frameIndex },
                                                set: { imageReferences[index].frameIndex = $0 }
                                            ),
                                            in: 0...240
                                        )
                                        Spacer()
                                        Text("Strength")
                                            .foregroundStyle(.secondary)
                                        TextField(
                                            "Strength",
                                            value: Binding(
                                                get: { imageReferences[index].strength },
                                                set: { imageReferences[index].strength = $0 }
                                            ),
                                            format: .number.precision(.fractionLength(2))
                                        )
                                        .frame(width: 80)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }

                        if needsVideoReference {
                            Button(selectedVideoURL == nil ? "Choose source video" : "Replace source video") {
                                isPickingVideo = true
                            }
                            .buttonStyle(.bordered)
                            if let selectedVideoURL {
                                HStack {
                                    Image(systemName: "film")
                                    Text(selectedVideoURL.lastPathComponent)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Button("Clear") {
                                        self.selectedVideoURL = nil
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                        }

                        if needsAudioReference {
                            Button(selectedAudioURL == nil ? "Choose audio reference" : "Replace audio reference") {
                                isPickingAudio = true
                            }
                            .buttonStyle(.bordered)
                            if let selectedAudioURL {
                                HStack {
                                    Image(systemName: "waveform")
                                    Text(selectedAudioURL.lastPathComponent)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Button("Clear") {
                                        self.selectedAudioURL = nil
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                        }

                        if needsLoRAReference {
                            Button(selectedLoRAURL == nil ? "Choose LoRA file" : "Replace LoRA file") {
                                isPickingLoRA = true
                            }
                            .buttonStyle(.bordered)
                            if let selectedLoRAURL {
                                HStack {
                                    Image(systemName: "cpu")
                                    Text(selectedLoRAURL.lastPathComponent)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Button("Clear") {
                                        self.selectedLoRAURL = nil
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                        }
                    }
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .animation(MLXRAnimation.spring, value: selectedTask)
                }

                // MARK: Output Parameters

                SectionCard(title: "Output", subtitle: "The first app keeps the meaningful motion controls visible rather than hiding them behind a future power-user panel.") {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 12) {
                        GridRow {
                            field("Width", value: $width)
                            field("Height", value: $height)
                            field("Frames", value: $numFrames)
                        }
                        GridRow {
                            field("FPS", value: $fps)
                            field("Steps", value: $numInferenceSteps)
                            field("Guidance", value: $guidanceScale, precision: .fractionLength(1))
                        }
                        GridRow {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Seed").foregroundStyle(.secondary)
                                TextField("Optional", text: $seed)
                                    .frame(width: 120)
                            }
                        }
                    }
                }

                // MARK: Advanced (family-local)

                if let selectedModel {
                    SectionCard(title: "Advanced", subtitle: "These controls are intentionally family-local and task-aware.") {
                        if !selectedModel.implementedPipelineVariants.isEmpty {
                            Picker("Pipeline variant", selection: $workflowVariant) {
                                ForEach(selectedModel.implementedPipelineVariants, id: \.self) { variant in
                                    Text(variant).tag(variant)
                                }
                            }
                            .pickerStyle(.segmented)
                        }

                        if selectedTask == .videoConditionVideo {
                            Picker("Control variant", selection: $controlVariant) {
                                Text("IC LoRA").tag("ic_lora")
                                Text("Motion Track").tag("motion_track_control")
                            }
                            .pickerStyle(.segmented)

                            field("Attention strength", value: $conditioningAttentionStrength, precision: .fractionLength(2))
                        }

                        if selectedTask == .videoRetake {
                            field("Window start (s)", value: $windowStartSeconds, precision: .fractionLength(2))
                            field("Window end (s)", value: $windowEndSeconds, precision: .fractionLength(2))
                            Toggle("Regenerate video", isOn: $regenerateVideo)
                            Toggle("Regenerate audio", isOn: $regenerateAudio)
                        }

                        if selectedModel.implementedPipelineVariants.isEmpty
                            && selectedTask != .videoConditionVideo
                            && selectedTask != .videoRetake
                        {
                            Text("This model family does not currently expose extra video-local controls.")
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                // MARK: Submit

                HStack(spacing: 14) {
                    Button {
                        Task { await submit() }
                    } label: {
                        HStack(spacing: 8) {
                            if isBusy {
                                ProgressView()
                                    .controlSize(.small)
                            }
                            Text(isBusy ? "Running\u{2026}" : "Run \(selectedTask.title)")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isBusy || selectedModel == nil || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    if let selectedModel {
                        Text("\(selectedModel.tasks.count) supported task(s) on this row")
                            .font(.system(.subheadline, design: .rounded))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(28)
        }
        .onAppear(perform: syncSelection)
        .onChange(of: selectedTask) { _, _ in
            syncSelection()
        }
        .fileImporter(
            isPresented: $isPickingImages,
            allowedContentTypes: [.image],
            allowsMultipleSelection: true
        ) { result in
            guard case let .success(urls) = result else { return }
            let startIndex = imageReferences.count
            imageReferences.append(contentsOf: urls.enumerated().map { offset, url in
                ImageReferenceDraft(fileURL: url, frameIndex: (startIndex + offset) * 20)
            })
        }
        .fileImporter(isPresented: $isPickingVideo, allowedContentTypes: [.movie]) { result in
            guard case let .success(url) = result else { return }
            selectedVideoURL = url
        }
        .fileImporter(isPresented: $isPickingAudio, allowedContentTypes: [.audio]) { result in
            guard case let .success(url) = result else { return }
            selectedAudioURL = url
        }
        .fileImporter(isPresented: $isPickingLoRA, allowedContentTypes: [.data]) { result in
            guard case let .success(url) = result else { return }
            selectedLoRAURL = url
        }
    }

    // MARK: - Computed Properties

    private var videoTasks: [ProductTask] {
        [
            .videoGenerate,
            .videoConditionImage,
            .videoConditionAudio,
            .videoConditionVideo,
            .videoInterpolate,
            .videoRetake,
        ]
    }

    private var availableModels: [ModelCatalogItem] {
        catalog.items(for: selectedTask)
    }

    private var selectedModel: ModelCatalogItem? {
        availableModels.first(where: { $0.modelId == selectedModelId }) ?? catalog.defaultModel(for: selectedTask)
    }

    private func installOperation(for modelId: String) -> ModelInstallOperationRecord? {
        installOperations.first { $0.modelId == modelId }
    }

    private var needsImageReferences: Bool {
        selectedTask == .videoConditionImage || selectedTask == .videoInterpolate
    }

    private var needsVideoReference: Bool {
        selectedTask == .videoConditionVideo || selectedTask == .videoRetake
    }

    private var needsAudioReference: Bool {
        selectedTask == .videoConditionAudio
    }

    private var needsLoRAReference: Bool {
        selectedTask == .videoConditionVideo
    }

    // MARK: - Helpers

    private func syncSelection() {
        if let defaultModel = catalog.defaultModel(for: selectedTask) {
            selectedModelId = defaultModel.modelId
            if workflowVariant.isEmpty {
                workflowVariant = defaultModel.implementedPipelineVariants.first ?? ""
            }
        }
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

    @ViewBuilder
    private func field(
        _ title: String,
        value: Binding<Double>,
        precision: FloatingPointFormatStyle<Double>.Configuration.Precision = .fractionLength(0)
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).foregroundStyle(.secondary)
            TextField(title, value: value, format: .number.precision(precision))
                .frame(width: 110)
        }
    }

    private func submit() async {
        guard let selectedModel else { return }

        var references: [MediaReferenceInput] = imageReferences.map {
            MediaReferenceInput(
                fileURL: $0.fileURL,
                kind: .image,
                metadata: [
                    "frame_index": .integer($0.frameIndex),
                    "strength": .number($0.strength),
                ]
            )
        }

        if let selectedVideoURL {
            references.append(
                MediaReferenceInput(
                    fileURL: selectedVideoURL,
                    kind: .video,
                    metadata: ["strength": .number(1.0)]
                )
            )
        }

        if let selectedAudioURL {
            references.append(
                MediaReferenceInput(
                    fileURL: selectedAudioURL,
                    kind: .audio,
                    metadata: [
                        "start_time_seconds": .number(0.0),
                        "max_duration_seconds": .null,
                    ]
                )
            )
        }

        if let selectedLoRAURL {
            references.append(
                MediaReferenceInput(
                    fileURL: selectedLoRAURL,
                    kind: .lora,
                    metadata: ["strength": .number(1.0)]
                )
            )
        }

        await onSubmit(
            VideoGenerationRequest(
                task: selectedTask,
                modelId: selectedModel.modelId,
                prompt: prompt,
                negativePrompt: negativePrompt,
                width: Int(width),
                height: Int(height),
                numFrames: Int(numFrames),
                fps: Int(fps),
                numInferenceSteps: Int(numInferenceSteps),
                guidanceScale: guidanceScale,
                seed: Int(seed),
                references: references,
                workflowVariant: workflowVariant.isEmpty ? nil : workflowVariant,
                controlVariant: selectedTask == .videoConditionVideo ? controlVariant : nil,
                conditioningAttentionStrength: selectedTask == .videoConditionVideo ? conditioningAttentionStrength : nil,
                windowStartSeconds: selectedTask == .videoRetake ? windowStartSeconds : nil,
                windowEndSeconds: selectedTask == .videoRetake ? windowEndSeconds : nil,
                regenerateVideo: regenerateVideo,
                regenerateAudio: regenerateAudio
            )
        )
    }
}

// MARK: - Image Reference Draft

private struct ImageReferenceDraft: Identifiable, Hashable {
    var id = UUID()
    var fileURL: URL
    var frameIndex: Int
    var strength: Double = 1.0
}
