import SwiftUI

// MARK: - Glass Card (replaces SectionCard)

public struct GlassCard<Content: View>: View {
    private let title: String?
    private let subtitle: String?
    private let content: Content

    public init(
        title: String? = nil,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.md) {
            if let title {
                VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                    Text(title)
                        .font(MLXRType.titleMedium)
                        .foregroundStyle(MLXRColor.textPrimary)
                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .font(MLXRType.bodyMedium)
                            .foregroundStyle(MLXRColor.textSecondary)
                    }
                }
            }
            content
        }
        .padding(MLXRSpacing.lg)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.07),
                            Color.white.opacity(0.03),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(0.12),
                                    Color.white.opacity(0.05),
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.5
                        )
                )
                .shadow(color: .black.opacity(0.15), radius: 16, y: 6)
        }
    }
}

// MARK: - Interactive Glass Card (hoverable)

public struct GlassCardInteractive<Content: View>: View {
    private let content: Content
    @State private var isHovered = false

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        content
            .padding(MLXRSpacing.lg)
            .background {
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(isHovered ? 0.09 : 0.07),
                                Color.white.opacity(isHovered ? 0.04 : 0.03),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: isHovered
                                        ? [MLXRColor.brandPrimary.opacity(0.4), MLXRColor.brandSecondary.opacity(0.2)]
                                        : [Color.white.opacity(0.12), Color.white.opacity(0.05)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: isHovered ? 1 : 0.5
                            )
                    )
                    .shadow(
                        color: isHovered ? MLXRColor.brandPrimary.opacity(0.1) : .black.opacity(0.08),
                        radius: isHovered ? 20 : 12,
                        y: isHovered ? 8 : 4
                    )
            }
            .scaleEffect(isHovered ? 1.015 : 1.0)
            .animation(MLXRMotion.cardHover, value: isHovered)
            .onHover { hovering in
                isHovered = hovering
            }
    }
}

// MARK: - Hero Card (cinematic header)

public struct HeroCard<Content: View>: View {
    private let title: String
    private let subtitle: String
    private let content: Content

    public init(
        title: String,
        subtitle: String,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
            Text(title)
                .font(MLXRType.displayMedium)
                .foregroundStyle(.white)
            Text(subtitle)
                .font(MLXRType.bodyLarge)
                .foregroundStyle(.white.opacity(0.72))
            content
        }
        .padding(MLXRSpacing.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                .fill(MLXRColor.heroGradient)
                .shadow(color: Color(red: 0.05, green: 0.10, blue: 0.25).opacity(0.30), radius: 20, y: 8)
        )
    }
}

// MARK: - Legacy Compatibility

public typealias SectionCard = GlassCard
public typealias HeroBanner = HeroCard
