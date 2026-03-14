import SwiftUI

// MARK: - Styled Text Field

public struct MLXRTextField: View {
    let label: String
    @Binding var text: String
    let placeholder: String

    public init(_ label: String, text: Binding<String>, placeholder: String = "") {
        self.label = label
        self._text = text
        self.placeholder = placeholder
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
                .tracking(0.8)

            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(MLXRType.bodyMedium)
                .padding(.horizontal, MLXRSpacing.md)
                .padding(.vertical, MLXRSpacing.sm)
                .background {
                    RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                        .fill(Color.white.opacity(0.05))
                        .overlay(
                            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                                .strokeBorder(
                                    LinearGradient(
                                        colors: [Color.white.opacity(0.1), Color.white.opacity(0.04)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 0.5
                                )
                        )
                }
        }
    }
}

// MARK: - Creative Control Segment

public struct CreativeControlSegment<T: Hashable>: View {
    let label: String
    @Binding var selection: T
    let options: [(T, String)]

    @Namespace private var segmentNamespace

    public init(label: String, selection: Binding<T>, options: [(T, String)]) {
        self.label = label
        self._selection = selection
        self.options = options
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
                .tracking(0.8)

            HStack(spacing: 2) {
                ForEach(options, id: \.0) { value, displayLabel in
                    Button {
                        withAnimation(MLXRMotion.snappy) {
                            selection = value
                        }
                    } label: {
                        Text(displayLabel)
                            .font(.system(size: 13, weight: selection == value ? .semibold : .medium, design: .rounded))
                            .foregroundStyle(selection == value ? .white : MLXRColor.textSecondary)
                            .padding(.horizontal, MLXRSpacing.md)
                            .padding(.vertical, 8)
                            .frame(maxWidth: .infinity)
                            .background {
                                if selection == value {
                                    Capsule(style: .continuous)
                                        .fill(MLXRColor.brandGradient)
                                        .shadow(color: MLXRColor.brandPrimary.opacity(0.3), radius: 6, y: 2)
                                        .matchedGeometryEffect(id: "segment", in: segmentNamespace)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(3)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(0.05))
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(Color.white.opacity(0.08), lineWidth: 0.5)
                    )
            )
        }
    }
}

// MARK: - Human-Readable Slider

public struct MLXRSlider: View {
    let label: String
    let leftLabel: String
    let rightLabel: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    public init(
        label: String,
        leftLabel: String,
        rightLabel: String,
        value: Binding<Double>,
        range: ClosedRange<Double> = 0...1
    ) {
        self.label = label
        self.leftLabel = leftLabel
        self.rightLabel = rightLabel
        self._value = value
        self.range = range
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
                .tracking(0.8)

            Slider(value: $value, in: range)
                .tint(MLXRColor.brandPrimary)

            HStack {
                Text(leftLabel)
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(MLXRColor.textTertiary)
                Spacer()
                Text(rightLabel)
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(MLXRColor.textTertiary)
            }
        }
    }
}

// MARK: - Prompt Composer

public struct PromptComposer: View {
    @Binding var text: String
    let placeholder: String
    let suggestions: [String]

    @State private var isFocused = false

    public init(
        text: Binding<String>,
        placeholder: String = "Describe your creation...",
        suggestions: [String] = []
    ) {
        self._text = text
        self.placeholder = placeholder
        self.suggestions = suggestions
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            ZStack(alignment: .topLeading) {
                if text.isEmpty {
                    Text(placeholder)
                        .font(MLXRType.bodyLarge)
                        .foregroundStyle(MLXRColor.textDisabled)
                        .padding(.horizontal, MLXRSpacing.md)
                        .padding(.vertical, MLXRSpacing.sm)
                }

                TextEditor(text: $text)
                    .font(MLXRType.bodyLarge)
                    .foregroundStyle(MLXRColor.textPrimary)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, MLXRSpacing.xs)
                    .padding(.vertical, MLXRSpacing.xxs)
                    .frame(minHeight: 80, maxHeight: 160)
            }
            .padding(MLXRSpacing.xs)
            .background {
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.06), Color.white.opacity(0.03)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [Color.white.opacity(0.12), Color.white.opacity(0.05)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 0.5
                            )
                    )
                    .shadow(color: .black.opacity(0.1), radius: 12, y: 4)
            }

            if !suggestions.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: MLXRSpacing.xs) {
                        ForEach(suggestions, id: \.self) { suggestion in
                            Button {
                                text = suggestion
                            } label: {
                                HStack(spacing: MLXRSpacing.xxs) {
                                    Image(systemName: "lightbulb.fill")
                                        .font(.system(size: 10))
                                    Text(suggestion)
                                        .lineLimit(1)
                                }
                                .font(MLXRType.captionLarge)
                                .foregroundStyle(MLXRColor.brandPrimary)
                                .padding(.horizontal, MLXRSpacing.sm)
                                .padding(.vertical, MLXRSpacing.xxs)
                                .background(
                                    Capsule(style: .continuous)
                                        .fill(MLXRColor.brandGlow)
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Reference Drop Zone

public struct ReferenceDropZone: View {
    let label: String
    let icon: String
    let isRequired: Bool
    let onPick: () -> Void
    let selectedFileName: String?

    @State private var isTargeted = false

    public init(
        label: String,
        icon: String = "photo.on.rectangle",
        isRequired: Bool = true,
        selectedFileName: String? = nil,
        onPick: @escaping () -> Void
    ) {
        self.label = label
        self.icon = icon
        self.isRequired = isRequired
        self.selectedFileName = selectedFileName
        self.onPick = onPick
    }

    public var body: some View {
        Button(action: onPick) {
            VStack(spacing: MLXRSpacing.sm) {
                if let fileName = selectedFileName {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 24))
                        .foregroundStyle(MLXRColor.brandSecondary)
                    Text(fileName)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(1)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 24, weight: .light))
                        .foregroundStyle(MLXRColor.textTertiary)
                    Text(label)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                    if !isRequired {
                        Text("Optional")
                            .font(MLXRType.captionSmall)
                            .foregroundStyle(MLXRColor.textTertiary)
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 120)
            .background {
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: isTargeted
                                ? [MLXRColor.brandPrimary.opacity(0.15), MLXRColor.brandPrimary.opacity(0.05)]
                                : [Color.white.opacity(0.05), Color.white.opacity(0.03)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .strokeBorder(
                                isTargeted ? MLXRColor.brandPrimary :
                                selectedFileName != nil ? MLXRColor.brandSecondary.opacity(0.3) :
                                MLXRColor.borderSubtle,
                                style: selectedFileName != nil ? .init() : StrokeStyle(lineWidth: 1, dash: [6, 4]),
                                antialiased: true
                            )
                    )
                    .shadow(color: .black.opacity(0.08), radius: 8, y: 3)
            }
            .animation(MLXRMotion.snappy, value: isTargeted)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Model Chip

public struct ModelChip: View {
    let name: String
    let status: ModelChipStatus

    public enum ModelChipStatus {
        case loaded, installed, installing, unavailable
    }

    public init(name: String, status: ModelChipStatus = .installed) {
        self.name = name
        self.status = status
    }

    public var body: some View {
        HStack(spacing: MLXRSpacing.xxs) {
            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)
            Text(name)
                .font(MLXRType.captionLarge)
                .foregroundStyle(MLXRColor.textSecondary)
                .lineLimit(1)
        }
        .padding(.horizontal, MLXRSpacing.sm)
        .padding(.vertical, MLXRSpacing.xxs)
        .background(
            Capsule(style: .continuous)
                .fill(MLXRColor.surfaceOverlay)
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
        )
    }

    private var statusColor: Color {
        switch status {
        case .loaded: MLXRColor.brandSecondary
        case .installed: MLXRColor.textTertiary
        case .installing: MLXRColor.brandWarm
        case .unavailable: MLXRColor.brandDanger
        }
    }
}
