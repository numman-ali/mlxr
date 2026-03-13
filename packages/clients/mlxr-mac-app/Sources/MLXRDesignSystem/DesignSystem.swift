import AppKit
import SwiftUI

// MARK: - Adaptive Color Theme

public enum MLXRTheme {

    // -- Brand accents (adapt to light / dark) --

    public static let accent = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.10, green: 0.48, blue: 0.96, alpha: 1),
        dark: .init(srgbRed: 0.38, green: 0.64, blue: 1.0, alpha: 1)
    ))

    public static let secondaryAccent = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.00, green: 0.73, blue: 0.66, alpha: 1),
        dark: .init(srgbRed: 0.24, green: 0.86, blue: 0.71, alpha: 1)
    ))

    public static let warmAccent = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.95, green: 0.55, blue: 0.18, alpha: 1),
        dark: .init(srgbRed: 1.0, green: 0.70, blue: 0.31, alpha: 1)
    ))

    public static let destructive = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.96, green: 0.26, blue: 0.21, alpha: 1),
        dark: .init(srgbRed: 1.0, green: 0.41, blue: 0.38, alpha: 1)
    ))

    // -- Surface & background --

    public static let surfacePrimary = Color(nsColor: adaptive(
        light: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.72),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.06)
    ))

    public static let surfaceSecondary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.03),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.04)
    ))

    public static let borderSubtle = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.08),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.10)
    ))

    // -- Hero gradient --

    public static let heroGradient = LinearGradient(
        colors: [
            Color(red: 0.07, green: 0.14, blue: 0.32),
            Color(red: 0.04, green: 0.32, blue: 0.52),
            Color(red: 0.08, green: 0.46, blue: 0.42),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    public static let backgroundGradientLight = LinearGradient(
        colors: [
            Color(red: 0.96, green: 0.97, blue: 1.0),
            Color(red: 0.93, green: 0.96, blue: 0.98),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    public static let backgroundGradientDark = LinearGradient(
        colors: [
            Color(red: 0.08, green: 0.09, blue: 0.12),
            Color(red: 0.06, green: 0.08, blue: 0.11),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    // -- Helpers --

    private static func adaptive(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        }
    }
}

// MARK: - Animation Presets

public enum MLXRAnimation {
    public static let spring = Animation.spring(response: 0.45, dampingFraction: 0.82)
    public static let snappy = Animation.spring(response: 0.3, dampingFraction: 0.78)
    public static let gentle = Animation.spring(response: 0.6, dampingFraction: 0.86)
    public static let quick = Animation.easeInOut(duration: 0.2)
}

// MARK: - Adaptive Background

public struct AdaptiveBackground: View {
    @Environment(\.colorScheme) private var colorScheme

    public init() {}

    public var body: some View {
        Group {
            if colorScheme == .dark {
                MLXRTheme.backgroundGradientDark
            } else {
                MLXRTheme.backgroundGradientLight
            }
        }
        .ignoresSafeArea()
    }
}

// MARK: - Feature Header

public struct FeatureHeader: View {
    private let eyebrow: String
    private let title: String
    private let subtitle: String

    public init(eyebrow: String, title: String, subtitle: String) {
        self.eyebrow = eyebrow
        self.title = title
        self.subtitle = subtitle
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(eyebrow.uppercased())
                .font(.system(.caption, design: .rounded, weight: .semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .contentTransition(.numericText())
            Text(subtitle)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Section Card

public struct SectionCard<Content: View>: View {
    private let title: String
    private let subtitle: String?
    private let content: Content

    public init(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(.title3, design: .rounded, weight: .semibold))
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.system(.subheadline, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
            content
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(MLXRTheme.borderSubtle, lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.04), radius: 8, y: 2)
        }
    }
}

// MARK: - Status Pill

public struct StatusPill: View {
    private let label: String
    private let tint: Color

    public init(label: String, tint: Color) {
        self.label = label
        self.tint = tint
    }

    public var body: some View {
        Text(label)
            .font(.system(.caption, design: .rounded, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(tint.opacity(0.14))
            )
            .foregroundStyle(tint)
    }
}

// MARK: - Metric Badge

public struct MetricBadge: View {
    private let label: String
    private let value: String

    public init(label: String, value: String) {
        self.label = label
        self.value = value
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.system(.headline, design: .rounded, weight: .semibold))
                .contentTransition(.numericText())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(MLXRTheme.surfaceSecondary)
        )
    }
}

// MARK: - Empty State View

public struct EmptyStateView: View {
    private let title: String
    private let subtitle: String
    private let systemImage: String

    public init(title: String, subtitle: String, systemImage: String) {
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
    }

    public var body: some View {
        VStack(spacing: 14) {
            Image(systemName: systemImage)
                .font(.system(size: 32, weight: .medium))
                .foregroundStyle(MLXRTheme.accent.opacity(0.7))
                .symbolEffect(.pulse.byLayer, options: .repeating.speed(0.5))
            Text(title)
                .font(.system(.title3, design: .rounded, weight: .semibold))
            Text(subtitle)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
        }
        .padding(28)
        .frame(maxWidth: .infinity)
        .background {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.thinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(MLXRTheme.borderSubtle, lineWidth: 0.5)
                )
        }
    }
}

// MARK: - Hero Banner

public struct HeroBanner<Content: View>: View {
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
        VStack(alignment: .leading, spacing: 18) {
            Text(title)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
            Text(subtitle)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.white.opacity(0.78))
            content
        }
        .padding(26)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(MLXRTheme.heroGradient)
                .shadow(color: Color(red: 0.07, green: 0.14, blue: 0.32).opacity(0.25), radius: 16, y: 6)
        )
    }
}

// MARK: - Inline Error Banner

public struct InlineErrorBanner: View {
    private let message: String
    private let onDismiss: (() -> Void)?

    public init(_ message: String, onDismiss: (() -> Void)? = nil) {
        self.message = message
        self.onDismiss = onDismiss
    }

    public var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MLXRTheme.destructive)
                .font(.system(size: 14, weight: .semibold))
            Text(message)
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.primary)
                .lineLimit(3)
            Spacer()
            if let onDismiss {
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(MLXRTheme.destructive.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(MLXRTheme.destructive.opacity(0.2), lineWidth: 0.5)
                )
        }
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: - Inline Success Banner

public struct InlineSuccessBanner: View {
    private let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(MLXRTheme.secondaryAccent)
                .font(.system(size: 14, weight: .semibold))
            Text(message)
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.primary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(MLXRTheme.secondaryAccent.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(MLXRTheme.secondaryAccent.opacity(0.2), lineWidth: 0.5)
                )
        }
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: - Progress Bar

public struct MLXRProgressBar: View {
    private let value: Double
    private let label: String?

    public init(value: Double, label: String? = nil) {
        self.value = min(max(value, 0), 1)
        self.label = label
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let label {
                HStack {
                    Text(label)
                        .font(.system(.caption, design: .rounded, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(value * 100))%")
                        .font(.system(.caption, design: .rounded, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .contentTransition(.numericText())
                }
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(MLXRTheme.surfaceSecondary)
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [MLXRTheme.accent, MLXRTheme.secondaryAccent],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * value)
                        .animation(MLXRAnimation.spring, value: value)
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())
        }
    }
}

// MARK: - Indeterminate Progress

public struct IndeterminateProgress: View {
    @State private var isAnimating = false

    private let label: String?

    public init(label: String? = nil) {
        self.label = label
    }

    public var body: some View {
        VStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            if let label {
                Text(label)
                    .font(.system(.caption, design: .rounded, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Detail Row

public struct DetailRow: View {
    private let label: String
    private let value: String

    public init(label: String, value: String) {
        self.label = label
        self.value = value
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased())
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.tertiary)
            Text(value)
                .font(.system(.subheadline, design: .rounded))
                .textSelection(.enabled)
        }
    }
}

// MARK: - Sidebar Item

public struct SidebarItem: View {
    private let title: String
    private let systemImage: String
    private let badge: String?
    private let isActive: Bool

    public init(title: String, systemImage: String, badge: String? = nil, isActive: Bool = false) {
        self.title = title
        self.systemImage = systemImage
        self.badge = badge
        self.isActive = isActive
    }

    public var body: some View {
        Label {
            HStack {
                Text(title)
                Spacer()
                if let badge {
                    Text(badge)
                        .font(.system(.caption2, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(MLXRTheme.warmAccent))
                        .transition(.scale.combined(with: .opacity))
                }
            }
        } icon: {
            Image(systemName: systemImage)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(isActive ? MLXRTheme.accent : .secondary)
        }
    }
}

// MARK: - Loading Card (shimmer placeholder)

public struct LoadingCard: View {
    @State private var shimmerOffset: CGFloat = -1

    public init() {}

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            shimmerRect(width: 180, height: 16)
            shimmerRect(width: 260, height: 12)
            shimmerRect(width: 220, height: 12)
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(MLXRTheme.borderSubtle, lineWidth: 0.5)
                )
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: false)) {
                shimmerOffset = 1
            }
        }
    }

    private func shimmerRect(width: CGFloat, height: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(MLXRTheme.surfaceSecondary)
            .frame(width: width, height: height)
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .fill(
                        LinearGradient(
                            colors: [.clear, .white.opacity(0.15), .clear],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .offset(x: shimmerOffset * width)
            )
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }
}
