import SwiftUI

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
            .font(MLXRType.captionSmall)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(tint.opacity(0.14))
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(tint.opacity(0.2), lineWidth: 0.5)
                    )
            )
            .foregroundStyle(tint)
            .shadow(color: tint.opacity(0.1), radius: 4, y: 1)
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
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .tracking(0.8)
                .foregroundStyle(MLXRColor.textTertiary)
            Text(value)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundStyle(
                    LinearGradient(
                        colors: [MLXRColor.textPrimary, MLXRColor.textSecondary],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .contentTransition(.numericText())
        }
        .padding(.horizontal, MLXRSpacing.md)
        .padding(.vertical, MLXRSpacing.sm)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.06), Color.white.opacity(0.02)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                )
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
                .shadow(color: .black.opacity(0.1), radius: 8, y: 3)
        }
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
        VStack(spacing: MLXRSpacing.lg) {
            ZStack {
                // Ambient glow
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                MLXRColor.brandPrimary.opacity(0.15),
                                MLXRColor.brandSecondary.opacity(0.05),
                                .clear,
                            ],
                            center: .center,
                            startRadius: 8,
                            endRadius: 48
                        )
                    )
                    .frame(width: 96, height: 96)

                Image(systemName: systemImage)
                    .font(.system(size: 40, weight: .light))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }

            VStack(spacing: MLXRSpacing.xs) {
                Text(title)
                    .font(MLXRType.titleLarge)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text(subtitle)
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
            }
        }
        .padding(MLXRSpacing.xxxl)
        .frame(maxWidth: .infinity)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.04), Color.white.opacity(0.01)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.1), Color.white.opacity(0.04)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.5
                        )
                )
                .shadow(color: .black.opacity(0.12), radius: 16, y: 6)
        }
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
        HStack(spacing: MLXRSpacing.xs) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MLXRColor.brandDanger)
                .font(.system(size: 14, weight: .semibold))
            Text(message)
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textPrimary)
                .lineLimit(3)
            Spacer()
            if let onDismiss {
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(MLXRColor.textTertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, MLXRSpacing.md)
        .padding(.vertical, MLXRSpacing.xs)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.sm, style: .continuous)
                .fill(MLXRColor.brandDanger.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.sm, style: .continuous)
                        .strokeBorder(MLXRColor.brandDanger.opacity(0.2), lineWidth: 0.5)
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
        HStack(spacing: MLXRSpacing.xs) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(MLXRColor.brandSecondary)
                .font(.system(size: 14, weight: .semibold))
            Text(message)
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textPrimary)
            Spacer()
        }
        .padding(.horizontal, MLXRSpacing.md)
        .padding(.vertical, MLXRSpacing.xs)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.sm, style: .continuous)
                .fill(MLXRColor.brandSecondary.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.sm, style: .continuous)
                        .strokeBorder(MLXRColor.brandSecondary.opacity(0.2), lineWidth: 0.5)
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
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            if let label {
                HStack {
                    Text(label)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textSecondary)
                    Spacer()
                    Text("\(Int(value * 100))%")
                        .font(MLXRType.captionSmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                        .contentTransition(.numericText())
                }
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(MLXRColor.surfaceOverlay)
                    Capsule()
                        .fill(MLXRColor.brandGradient)
                        .frame(width: geometry.size.width * value)
                        .animation(MLXRMotion.spring, value: value)
                        .shadow(color: MLXRColor.brandPrimary.opacity(0.3), radius: 4, y: 0)
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())
        }
    }
}

// MARK: - Indeterminate Progress

public struct IndeterminateProgress: View {
    private let label: String?

    public init(label: String? = nil) {
        self.label = label
    }

    public var body: some View {
        VStack(spacing: MLXRSpacing.xs) {
            ProgressView()
                .controlSize(.small)
            if let label {
                Text(label)
                    .font(MLXRType.captionLarge)
                    .foregroundStyle(MLXRColor.textSecondary)
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
        VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
            Text(label.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            Text(value)
                .font(MLXRType.bodyMedium)
                .foregroundStyle(MLXRColor.textPrimary)
                .textSelection(.enabled)
        }
    }
}

// MARK: - Loading Card

public struct LoadingCard: View {
    public init() {}

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.md) {
            placeholderRect(width: 180, height: 16)
            placeholderRect(width: 260, height: 12)
            placeholderRect(width: 220, height: 12)
        }
        .padding(MLXRSpacing.lg)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
        }
    }

    private func placeholderRect(width: CGFloat, height: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(MLXRColor.surfaceOverlay)
            .frame(width: width, height: height)
    }
}

// MARK: - Completion Toast

public struct CompletionToast: View {
    let message: String
    let onView: (() -> Void)?

    public init(message: String, onView: (() -> Void)? = nil) {
        self.message = message
        self.onView = onView
    }

    public var body: some View {
        HStack(spacing: MLXRSpacing.sm) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(MLXRColor.brandSecondary)
                .font(.system(size: 16, weight: .semibold))
            Text(message)
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textPrimary)
                .lineLimit(1)
            Spacer()
            if let onView {
                Button("View", action: onView)
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(MLXRColor.brandPrimary)
                    .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, MLXRSpacing.md)
        .padding(.vertical, MLXRSpacing.xs)
        .background(
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(MLXRColor.brandSecondary.opacity(0.3), lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
        )
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}

// MARK: - Adaptive Background

public struct AdaptiveBackground: View {
    public init() {}

    public var body: some View {
        ZStack {
            MLXRColor.canvasDeep
                .ignoresSafeArea()

            // Subtle radial ambient light from top-left — cinematic depth
            RadialGradient(
                colors: [
                    Color(red: 0.08, green: 0.12, blue: 0.22).opacity(0.4),
                    .clear,
                ],
                center: .topLeading,
                startRadius: 100,
                endRadius: 600
            )
            .ignoresSafeArea()

            // Subtle warm accent from bottom-right
            RadialGradient(
                colors: [
                    Color(red: 0.12, green: 0.08, blue: 0.18).opacity(0.25),
                    .clear,
                ],
                center: .bottomTrailing,
                startRadius: 50,
                endRadius: 500
            )
            .ignoresSafeArea()
        }
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
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(eyebrow.uppercased())
                .font(MLXRType.captionSmall)
                .tracking(2)
                .foregroundStyle(
                    LinearGradient(
                        colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
            Text(title)
                .font(MLXRType.displayMedium)
                .foregroundStyle(MLXRColor.textPrimary)
                .contentTransition(.numericText())
            Text(subtitle)
                .font(MLXRType.bodyLarge)
                .foregroundStyle(MLXRColor.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Sidebar Item (legacy compat, used by old modules during transition)

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
                        .font(MLXRType.captionSmall)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(MLXRColor.brandWarm))
                        .transition(.scale.combined(with: .opacity))
                }
            }
        } icon: {
            Image(systemName: systemImage)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(isActive ? MLXRColor.brandPrimary : .secondary)
        }
    }
}
