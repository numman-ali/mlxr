import SwiftUI

// MARK: - Nav Rail

public struct NavRail<D: Hashable & CaseIterable & Identifiable>: View
    where D.AllCases: RandomAccessCollection
{
    @Binding var selection: D?
    let destinations: [D]
    let items: (D) -> NavRailItemConfig
    let footer: AnyView?

    public init(
        selection: Binding<D?>,
        destinations: [D] = Array(D.allCases),
        items: @escaping (D) -> NavRailItemConfig,
        footer: AnyView? = nil
    ) {
        self._selection = selection
        self.destinations = destinations
        self.items = items
        self.footer = footer
    }

    public var body: some View {
        VStack(spacing: 0) {
            // MLXR brand glyph — cinematic logo
            VStack(spacing: 6) {
                ZStack {
                    Circle()
                        .fill(Color.white.opacity(0.03))
                        .frame(width: 54, height: 54)
                        .overlay(
                            Circle()
                                .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                        )

                    Image(systemName: "sparkles")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }

                Text("MLXR")
                    .font(.system(size: 11, weight: .heavy, design: .rounded))
                    .tracking(2)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [MLXRColor.textSecondary, MLXRColor.textTertiary],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
            }
            .padding(.top, MLXRSpacing.lg)
            .padding(.bottom, MLXRSpacing.xl)

            // Destination items
            VStack(spacing: MLXRSpacing.xs) {
                ForEach(destinations) { destination in
                    let config = items(destination)
                    NavRailButton(
                        icon: config.icon,
                        label: config.label,
                        badge: config.badge,
                        isActive: selection == destination
                    ) {
                        withAnimation(MLXRMotion.snappy) {
                            selection = destination
                        }
                    }
                }
            }

            Spacer()

            if let footer {
                footer
                    .padding(.bottom, MLXRSpacing.lg)
            }
        }
        .frame(width: 76)
        .background {
            ZStack {
                MLXRColor.canvasDeep
                // Subtle vertical gradient for depth
                LinearGradient(
                    colors: [
                        Color.white.opacity(0.02),
                        .clear,
                        Color.white.opacity(0.01),
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            }
        }
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [
                            MLXRColor.brandPrimary.opacity(0.15),
                            MLXRColor.borderSubtle,
                            MLXRColor.brandSecondary.opacity(0.1),
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 0.5)
        }
    }
}

// MARK: - Nav Rail Item Config

public struct NavRailItemConfig {
    public let icon: String
    public let label: String
    public let badge: String?

    public init(icon: String, label: String, badge: String? = nil) {
        self.icon = icon
        self.label = label
        self.badge = badge
    }
}

// MARK: - Nav Rail Button

public struct NavRailButton: View {
    let icon: String
    let label: String
    let badge: String?
    let isActive: Bool
    let action: () -> Void

    @State private var isHovered = false

    public init(icon: String, label: String, badge: String? = nil, isActive: Bool, action: @escaping () -> Void) {
        self.icon = icon
        self.label = label
        self.badge = badge
        self.isActive = isActive
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            VStack(spacing: 5) {
                ZStack(alignment: .topTrailing) {
                    ZStack {
                        // Active glow ring
                        if isActive {
                            Circle()
                                .fill(
                                    RadialGradient(
                                        colors: [
                                            MLXRColor.brandPrimary.opacity(0.18),
                                            MLXRColor.brandPrimary.opacity(0.03),
                                            .clear,
                                        ],
                                        center: .center,
                                        startRadius: 2,
                                        endRadius: 18
                                    )
                                )
                                .frame(width: 36, height: 36)
                        }

                        Image(systemName: icon)
                            .font(.system(size: 20, weight: isActive ? .semibold : .regular))
                            .foregroundStyle(
                                isActive
                                    ? AnyShapeStyle(LinearGradient(
                                        colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary.opacity(0.8)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ))
                                    : AnyShapeStyle(isHovered ? MLXRColor.textSecondary : MLXRColor.textTertiary)
                            )
                            .frame(width: 36, height: 32)
                    }

                    if let badge {
                        Text(badge)
                            .font(.system(size: 9, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(
                                Capsule()
                                    .fill(MLXRColor.brandWarm)
                                    .shadow(color: MLXRColor.brandWarm.opacity(0.18), radius: 2, y: 1)
                            )
                            .offset(x: 8, y: -6)
                            .transition(.scale.combined(with: .opacity))
                    }
                }

                Text(label)
                    .font(.system(size: 10, weight: isActive ? .bold : .medium, design: .rounded))
                    .foregroundStyle(isActive ? MLXRColor.brandPrimary : (isHovered ? MLXRColor.textSecondary : MLXRColor.textTertiary))
            }
            .frame(width: 60, height: 56)
            .background(
                RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                    .fill(
                        isActive ? MLXRColor.brandGlow :
                        isHovered ? MLXRColor.surfaceHover :
                        .clear
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                            .strokeBorder(
                                isActive ? MLXRColor.brandPrimary.opacity(0.25) : .clear,
                                lineWidth: 0.5
                            )
                    )
            )
            .animation(MLXRMotion.snappy, value: isActive)
            .animation(MLXRMotion.micro, value: isHovered)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovered = hovering
        }
    }
}

// MARK: - Quick Create Bar

public struct QuickCreateBar: View {
    @Binding var text: String
    let placeholder: String
    let onSubmit: () -> Void

    @State private var isFocused = false

    public init(text: Binding<String>, placeholder: String = "Describe what you want to create...", onSubmit: @escaping () -> Void) {
        self._text = text
        self.placeholder = placeholder
        self.onSubmit = onSubmit
    }

    public var body: some View {
        HStack(spacing: MLXRSpacing.md) {
            Image(systemName: "sparkles")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(
                    LinearGradient(
                        colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(MLXRType.bodyLarge)
                .foregroundStyle(MLXRColor.textPrimary)
                .onSubmit(onSubmit)

            if !text.isEmpty {
                Image(systemName: "arrow.right.circle.fill")
                    .font(.system(size: 20))
                    .foregroundStyle(MLXRColor.brandPrimary)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.vertical, 14)
        .background {
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(
                            isFocused
                                ? LinearGradient(
                                    colors: [MLXRColor.brandPrimary.opacity(0.6), MLXRColor.brandSecondary.opacity(0.4)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                                : LinearGradient(
                                    colors: [MLXRColor.borderSubtle, MLXRColor.borderSubtle],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                            lineWidth: isFocused ? 1.5 : 0.5
                        )
                )
                .shadow(color: MLXRColor.brandPrimary.opacity(isFocused ? 0.15 : 0), radius: 16, y: 0)
                .shadow(color: .black.opacity(0.12), radius: 12, y: 4)
        }
        .animation(MLXRMotion.snappy, value: isFocused)
        .animation(MLXRMotion.snappy, value: text.isEmpty)
    }
}
